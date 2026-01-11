"""
Core simulation engine for direct indexing backtests and forward tests.

Manages portfolio state, executes trades, and handles rebalancing and TLH logic.
"""

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from decimal import Decimal, ROUND_DOWN, ROUND_HALF_UP
from enum import Enum
from typing import Optional
import uuid

import pandas as pd

from di_pilot.models import (
    BenchmarkConstituent,
    PriceData,
    TaxLot,
    TradeSide,
    TradeRationale,
    GainType,
)

from di_pilot.data.sectors import calculate_sector_weights


class TradeReason(Enum):
    """Reason for a simulated trade."""
    INITIAL_PURCHASE = "INITIAL_PURCHASE"
    REBALANCE_BUY = "REBALANCE_BUY"
    REBALANCE_SELL = "REBALANCE_SELL"
    TLH_SELL = "TLH_SELL"
    TLH_REPLACEMENT_BUY = "TLH_REPLACEMENT_BUY"
    DIVIDEND = "DIVIDEND"


@dataclass
class SimulationTrade:
    """Record of a simulated trade."""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: TradeSide
    shares: Decimal
    price: Decimal
    value: Decimal
    reason: TradeReason
    lot_id: Optional[str] = None  # For sells, which lot was sold
    notes: str = ""

    @classmethod
    def create(
        cls,
        timestamp: datetime,
        symbol: str,
        side: TradeSide,
        shares: Decimal,
        price: Decimal,
        reason: TradeReason,
        lot_id: Optional[str] = None,
        notes: str = "",
    ) -> "SimulationTrade":
        return cls(
            trade_id=str(uuid.uuid4()),
            timestamp=timestamp,
            symbol=symbol,
            side=side,
            shares=shares,
            price=price,
            value=shares * price,
            reason=reason,
            lot_id=lot_id,
            notes=notes,
        )


@dataclass
class DailySnapshot:
    """Daily portfolio snapshot."""
    date: date
    total_value: Decimal
    cash: Decimal
    holdings_value: Decimal
    num_positions: int
    num_lots: int
    unrealized_pnl: Decimal
    realized_pnl: Decimal
    daily_return: Decimal
    cumulative_return: Decimal
    sector_weights: dict[str, float] = field(default_factory=dict)  # Added for sector drift tracking


@dataclass
class SimulationConfig:
    """Configuration for simulation runs."""
    initial_cash: Decimal = Decimal("1000000")
    cash_buffer_pct: Decimal = Decimal("0.01")  # Keep 1% cash
    min_trade_value: Decimal = Decimal("100")
    max_turnover_pct: Decimal = Decimal("0.10")  # Max 10% turnover per rebalance
    rebalance_band_pct: Decimal = Decimal("0.02")  # Rebalance if drift > 2%
    tlh_loss_threshold: Decimal = Decimal("0.05")  # Harvest losses > 5% (institutional standard)
    tlh_wash_sale_days: int = 30  # Wash sale window
    use_current_constituents: bool = True  # Mode 1 vs Mode 2
    rebalance_freq: str = "weekly"  # weekly, monthly, quarterly

    # Fidelity-style Direct Indexing settings
    # Philosophy: Hold ALL index constituents for accurate tracking
    target_positions: int = 0  # 0 means hold ALL S&P 500 constituents (~503 stocks)
    # Set to 350 for "institutional" approach with fewer positions
    max_tlh_trades_per_day: int = 10  # Conservative: max 10 TLH trades per day
    max_tlh_pct_per_day: Decimal = Decimal("0.03")  # Max 3% of portfolio harvested per day
    
    # TLH behavior mode
    # Professional direct indexers (Parametric, Aperio) do IMMEDIATE replacement buys
    # to maintain market exposure. This is the recommended approach.
    # If True: TLH sells just add cash (NOT recommended - creates cash drag)
    # If False: TLH sells are immediately followed by correlated replacement buys
    tlh_skip_replacement_buys: bool = False  # Professional behavior: immediate replacements
    
    # Dividend simulation (matching Fidelity pattern)
    # If True: Simulate dividend payments based on yield estimates
    enable_dividend_simulation: bool = True  # Enable by default to match real-world behavior
    average_dividend_yield: Decimal = Decimal("0.015")  # 1.5% average S&P yield


@dataclass
class SimulationState:
    """Current state of a simulation."""
    portfolio_id: str
    current_date: date
    cash: Decimal
    lots: list[TaxLot] = field(default_factory=list)
    trades: list[SimulationTrade] = field(default_factory=list)
    snapshots: list[DailySnapshot] = field(default_factory=list)
    realized_pnl: Decimal = Decimal("0")
    harvested_losses: Decimal = Decimal("0")
    wash_sale_symbols: dict[str, date] = field(default_factory=dict)  # symbol -> sell date

    @property
    def total_cost_basis(self) -> Decimal:
        """Total cost basis of all lots."""
        return sum(lot.shares * lot.cost_basis for lot in self.lots)

    def get_lots_by_symbol(self, symbol: str) -> list[TaxLot]:
        """Get all lots for a symbol."""
        return [lot for lot in self.lots if lot.symbol == symbol.upper()]

    def get_total_shares(self, symbol: str) -> Decimal:
        """Get total shares for a symbol."""
        return sum(lot.shares for lot in self.get_lots_by_symbol(symbol))

    def is_wash_sale_restricted(self, symbol: str, current_date: date, window_days: int = 30) -> bool:
        """Check if symbol is in wash sale restriction window."""
        if symbol not in self.wash_sale_symbols:
            return False
        sell_date = self.wash_sale_symbols[symbol]
        days_since_sell = (current_date - sell_date).days
        return days_since_sell < window_days


class SimulationEngine:
    """
    Core simulation engine.

    Manages portfolio state transitions for backtesting and forward testing.
    """

    def __init__(self, config: SimulationConfig):
        """
        Initialize simulation engine.

        Args:
            config: Simulation configuration
        """
        self.config = config

    def initialize_portfolio(
        self,
        portfolio_id: str,
        start_date: date,
        constituents: list[BenchmarkConstituent],
        prices: dict[str, PriceData],
    ) -> SimulationState:
        """
        Initialize portfolio from cash.

        Uses institutional direct indexing approach:
        - Selects top N constituents by weight (default 350)
        - Maintains tight tracking error (<0.3%) while enabling efficient TLH

        Args:
            portfolio_id: Unique identifier for this simulation
            start_date: Start date for the simulation
            constituents: Benchmark constituents with target weights
            prices: Current prices by symbol

        Returns:
            Initial SimulationState with holdings
        """
        state = SimulationState(
            portfolio_id=portfolio_id,
            current_date=start_date,
            cash=self.config.initial_cash,
        )

        # Calculate investable cash (after buffer)
        investable = self.config.initial_cash * (Decimal("1") - self.config.cash_buffer_pct)

        # Filter constituents to those with prices
        valid_constituents = [
            c for c in constituents
            if c.symbol in prices and prices[c.symbol].close > Decimal("0")
        ]

        if not valid_constituents:
            raise ValueError("No valid constituents with prices")

        # Institutional approach: select top N by weight for tight tracking + efficient TLH
        if self.config.target_positions and self.config.target_positions > 0:
            # Sort by weight descending
            valid_constituents = sorted(valid_constituents, key=lambda c: c.weight, reverse=True)
            # Take top N positions
            valid_constituents = valid_constituents[:self.config.target_positions]

        # Renormalize weights to sum to 1.0
        total_weight = sum(c.weight for c in valid_constituents)
        if total_weight == Decimal("0"):
            raise ValueError("Total weight is zero")

        # Allocate to each constituent
        for constituent in valid_constituents:
            normalized_weight = constituent.weight / total_weight
            target_value = investable * normalized_weight

            if target_value < self.config.min_trade_value:
                continue

            price = prices[constituent.symbol].close
            shares = (target_value / price).quantize(
                Decimal("0.000001"), rounding=ROUND_DOWN
            )

            if shares <= Decimal("0"):
                continue

            actual_cost = shares * price

            # Create lot
            lot = TaxLot.create(
                symbol=constituent.symbol,
                shares=shares,
                cost_basis=price,
                acquisition_date=start_date,
                portfolio_id=portfolio_id,
            )
            state.lots.append(lot)

            # Record trade
            trade = SimulationTrade.create(
                timestamp=datetime.combine(start_date, datetime.min.time()),
                symbol=constituent.symbol,
                side=TradeSide.BUY,
                shares=shares,
                price=price,
                reason=TradeReason.INITIAL_PURCHASE,
                notes=f"Initial allocation at {normalized_weight:.4%} weight",
            )
            state.trades.append(trade)

            # Update cash
            state.cash -= actual_cost

        return state

    def value_portfolio(
        self,
        state: SimulationState,
        prices: dict[str, PriceData],
    ) -> Decimal:
        """
        Calculate current portfolio value.

        Args:
            state: Current simulation state
            prices: Current prices by symbol

        Returns:
            Total portfolio value (holdings + cash)
        """
        holdings_value = Decimal("0")
        for lot in state.lots:
            if lot.symbol in prices:
                holdings_value += lot.shares * prices[lot.symbol].close

        return holdings_value + state.cash

    def calculate_weights(
        self,
        state: SimulationState,
        prices: dict[str, PriceData],
    ) -> dict[str, Decimal]:
        """
        Calculate current portfolio weights by symbol.

        Args:
            state: Current simulation state
            prices: Current prices

        Returns:
            Dictionary mapping symbol to weight
        """
        total_value = self.value_portfolio(state, prices)
        if total_value == Decimal("0"):
            return {}

        weights = {}
        for lot in state.lots:
            symbol = lot.symbol
            if symbol in prices:
                lot_value = lot.shares * prices[symbol].close
                weights[symbol] = weights.get(symbol, Decimal("0")) + lot_value / total_value

        return weights

    def calculate_target_weights(
        self,
        constituents: list[BenchmarkConstituent],
        prices: dict[str, PriceData],
        base_prices: dict[str, PriceData],
    ) -> dict[str, Decimal]:
        """
        Calculate price-adjusted target weights for market-cap tracking.
        
        This mimics how SPY's weights change daily as prices move:
        - Stocks that go up get higher weights
        - Stocks that go down get lower weights
        
        The formula assumes shares outstanding are constant, so:
        new_weight = old_weight × (new_price / old_price) / normalization_factor
        
        Args:
            constituents: Base constituent weights (from Day 1)
            prices: Current prices
            base_prices: Prices from the initial date
            
        Returns:
            Dictionary of price-adjusted target weights
        """
        adjusted_weights = {}
        total_adjusted = Decimal("0")
        
        for c in constituents:
            symbol = c.symbol
            if symbol not in prices or symbol not in base_prices:
                continue
            
            current_price = prices[symbol].close
            initial_price = base_prices[symbol].close
            
            if initial_price <= 0:
                continue
            
            # Adjust weight by price change ratio
            # This approximates market-cap weighting assuming constant shares outstanding
            price_ratio = current_price / initial_price
            adjusted_weight = c.weight * price_ratio
            adjusted_weights[symbol] = adjusted_weight
            total_adjusted += adjusted_weight
        
        # Normalize to sum to 1.0
        if total_adjusted > 0:
            adjusted_weights = {
                s: w / total_adjusted for s, w in adjusted_weights.items()
            }
        
        return adjusted_weights


    def calculate_drift(
        self,
        state: SimulationState,
        constituents: list[BenchmarkConstituent],
        prices: dict[str, PriceData],
        base_prices: dict[str, PriceData] = None,
    ) -> dict[str, Decimal]:
        """
        Calculate drift from target weights.

        Args:
            state: Current simulation state
            constituents: Target benchmark weights
            prices: Current prices
            base_prices: Optional initial prices for price-adjusted weighting.
                         When provided, uses dynamic market-cap-style weights.

        Returns:
            Dictionary mapping symbol to drift (current - target)
        """
        current_weights = self.calculate_weights(state, prices)
        
        # Use price-adjusted weights if base_prices provided (professional mode)
        if base_prices:
            target_weights = self.calculate_target_weights(constituents, prices, base_prices)
        else:
            target_weights = {c.symbol: c.weight for c in constituents}

        # Normalize target weights to constituents we hold
        held_symbols = set(current_weights.keys())
        target_sum = sum(
            w for s, w in target_weights.items()
            if s in held_symbols or s in prices
        )

        if target_sum > 0:
            target_weights = {
                s: w / target_sum for s, w in target_weights.items()
            }

        drift = {}
        all_symbols = set(current_weights.keys()) | set(target_weights.keys())

        for symbol in all_symbols:
            current = current_weights.get(symbol, Decimal("0"))
            target = target_weights.get(symbol, Decimal("0"))
            drift[symbol] = current - target

        return drift

    def identify_tlh_candidates(
        self,
        state: SimulationState,
        prices: dict[str, PriceData],
        current_date: date,
    ) -> list[TaxLot]:
        """
        Identify lots eligible for tax-loss harvesting.

        Args:
            state: Current simulation state
            prices: Current prices
            current_date: Current date for holding period calculation

        Returns:
            List of lots with losses exceeding threshold
        """
        candidates = []

        for lot in state.lots:
            if lot.symbol not in prices:
                continue

            current_price = prices[lot.symbol].close
            cost_basis = lot.cost_basis
            pnl_pct = (current_price - cost_basis) / cost_basis

            # Check if loss exceeds threshold
            if pnl_pct < -self.config.tlh_loss_threshold:
                candidates.append(lot)

        # Sort by loss magnitude (most negative first)
        candidates.sort(
            key=lambda lot: (prices[lot.symbol].close - lot.cost_basis) / lot.cost_basis
        )

        return candidates

    def execute_rebalance(
        self,
        state: SimulationState,
        constituents: list[BenchmarkConstituent],
        prices: dict[str, PriceData],
        current_date: date,
        base_prices: dict[str, PriceData] = None,
    ) -> SimulationState:
        """
        Execute rebalancing trades.

        Args:
            state: Current simulation state
            constituents: Target benchmark weights
            prices: Current prices
            current_date: Current date
            base_prices: Optional initial prices for price-adjusted weighting

        Returns:
            Updated simulation state
        """
        drift = self.calculate_drift(state, constituents, prices, base_prices)
        total_value = self.value_portfolio(state, prices)
        max_trade_value = total_value * self.config.max_turnover_pct

        traded_value = Decimal("0")
        timestamp = datetime.combine(current_date, datetime.min.time())

        # Process sells first (overweight positions)
        for symbol, symbol_drift in sorted(drift.items(), key=lambda x: x[1], reverse=True):
            if symbol_drift <= self.config.rebalance_band_pct:
                continue
            if symbol not in prices:
                continue

            # Calculate shares to sell
            trade_value = min(
                abs(symbol_drift) * total_value,
                max_trade_value - traded_value,
            )

            if trade_value < self.config.min_trade_value:
                continue

            price = prices[symbol].close
            shares_to_sell = (trade_value / price).quantize(
                Decimal("0.000001"), rounding=ROUND_DOWN
            )

            if shares_to_sell <= Decimal("0"):
                continue

            # Sell from lots (FIFO)
            remaining = shares_to_sell
            lots_to_sell = state.get_lots_by_symbol(symbol)
            lots_to_sell.sort(key=lambda l: l.acquisition_date)

            for lot in lots_to_sell:
                if remaining <= Decimal("0"):
                    break

                sell_shares = min(remaining, lot.shares)
                remaining -= sell_shares

                # Record realized P&L
                realized = sell_shares * (price - lot.cost_basis)
                state.realized_pnl += realized

                # Record trade
                trade = SimulationTrade.create(
                    timestamp=timestamp,
                    symbol=symbol,
                    side=TradeSide.SELL,
                    shares=sell_shares,
                    price=price,
                    reason=TradeReason.REBALANCE_SELL,
                    lot_id=lot.lot_id,
                    notes=f"Rebalance sell, drift {symbol_drift:.4%}",
                )
                state.trades.append(trade)

                # Update cash
                state.cash += sell_shares * price
                traded_value += sell_shares * price

                # Update lot
                lot.shares -= sell_shares
                if lot.shares <= Decimal("0"):
                    state.lots.remove(lot)

        # Process buys (underweight positions)
        for symbol, symbol_drift in sorted(drift.items(), key=lambda x: x[1]):
            if symbol_drift >= -self.config.rebalance_band_pct:
                continue
            if symbol not in prices:
                continue

            # Calculate shares to buy
            trade_value = min(
                abs(symbol_drift) * total_value,
                max_trade_value - traded_value,
                state.cash * (Decimal("1") - self.config.cash_buffer_pct),
            )

            if trade_value < self.config.min_trade_value:
                continue

            price = prices[symbol].close
            shares_to_buy = (trade_value / price).quantize(
                Decimal("0.000001"), rounding=ROUND_DOWN
            )

            if shares_to_buy <= Decimal("0"):
                continue

            # Create new lot
            lot = TaxLot.create(
                symbol=symbol,
                shares=shares_to_buy,
                cost_basis=price,
                acquisition_date=current_date,
                portfolio_id=state.portfolio_id,
            )
            state.lots.append(lot)

            # Record trade
            trade = SimulationTrade.create(
                timestamp=timestamp,
                symbol=symbol,
                side=TradeSide.BUY,
                shares=shares_to_buy,
                price=price,
                reason=TradeReason.REBALANCE_BUY,
                notes=f"Rebalance buy, drift {symbol_drift:.4%}",
            )
            state.trades.append(trade)

            # Update cash
            state.cash -= shares_to_buy * price
            traded_value += shares_to_buy * price

        return state

    def execute_tlh(
        self,
        state: SimulationState,
        prices: dict[str, PriceData],
        current_date: date,
    ) -> SimulationState:
        """
        Execute tax-loss harvesting with correlated replacement buys.

        For each loss candidate:
        1. Sell the losing position
        2. Buy a correlated replacement (same sector, different company)
        3. Maintain market exposure while realizing tax loss

        Includes safeguards against:
        - Ping-pong trading (A→B, B→A same day)
        - Over-harvesting (daily limits)
        - Cash accumulation (proper rounding)

        Args:
            state: Current simulation state
            prices: Current prices
            current_date: Current date

        Returns:
            Updated simulation state
        """
        from di_pilot.data.tlh_pairs import get_replacement_candidates, get_sector_etf_fallback

        # TLH limits from config - institutional approach: index tracking first, TLH second
        max_trades = self.config.max_tlh_trades_per_day
        max_harvest_pct = self.config.max_tlh_pct_per_day

        candidates = self.identify_tlh_candidates(state, prices, current_date)
        timestamp = datetime.combine(current_date, datetime.min.time())

        # Build set of wash-sale restricted symbols (from previous days)
        wash_restricted = {
            symbol for symbol, sell_date in state.wash_sale_symbols.items()
            if (current_date - sell_date).days < self.config.tlh_wash_sale_days
        }

        # FIX 1: Track symbols traded TODAY to prevent ping-pong (A→B, B→A)
        traded_today: set[str] = set()

        # Track TLH activity for limits
        tlh_trades_today = 0
        harvested_today = Decimal("0")

        # Calculate portfolio value for harvest limit
        portfolio_value = state.cash + sum(
            lot.shares * prices.get(lot.symbol, PriceData(lot.symbol, current_date, lot.cost_basis)).close
            for lot in state.lots
        )

        # FIX 2: Get existing portfolio symbols to avoid buying what we already own
        existing_symbols = {lot.symbol for lot in state.lots}

        for lot in candidates:
            # Check TLH limits (index tracking first, TLH second)
            if tlh_trades_today >= max_trades:
                break
            if harvested_today >= portfolio_value * max_harvest_pct:
                break

            # Check wash sale restriction
            if state.is_wash_sale_restricted(
                lot.symbol, current_date, self.config.tlh_wash_sale_days
            ):
                continue

            # FIX 1: Skip if this symbol was already traded today (prevents ping-pong)
            if lot.symbol in traded_today:
                continue

            price = prices[lot.symbol].close
            loss_amount = lot.shares * (price - lot.cost_basis)
            proceeds = lot.shares * price

            # MODE: Skip replacement buys (Fidelity pattern)
            # Just sell the losing position and let cash accumulate for rebalancing
            if self.config.tlh_skip_replacement_buys:
                # Remove the lot
                state.lots.remove(lot)
                
                # Record TLH sell trade
                sell_trade = SimulationTrade.create(
                    timestamp=timestamp,
                    symbol=lot.symbol,
                    side=TradeSide.SELL,
                    shares=lot.shares,
                    price=price,
                    reason=TradeReason.TLH_SELL,
                    lot_id=lot.lot_id,
                    notes=f"TLH harvest (no replacement), loss ${abs(loss_amount):.2f}",
                )
                state.trades.append(sell_trade)
                
                # Add proceeds to cash
                state.cash += proceeds
                
                # Update tracking
                state.realized_pnl += loss_amount
                state.harvested_losses += abs(loss_amount)
                state.wash_sale_symbols[lot.symbol] = current_date
                traded_today.add(lot.symbol)
                
                # Track limits
                tlh_trades_today += 1
                harvested_today += abs(loss_amount)
                continue

            # MODE: Do replacement buys (original behavior)
            # Find a valid replacement before proceeding
            # FIX 1: Exclude both wash_restricted AND traded_today from candidates
            excluded_symbols = wash_restricted | traded_today | existing_symbols
            replacement_candidates = get_replacement_candidates(lot.symbol, excluded_symbols)

            replacement_symbol = None
            for candidate in replacement_candidates:
                # FIX 2: Skip if we already own this symbol
                if candidate in existing_symbols:
                    continue
                if candidate in prices and prices[candidate].close > Decimal("0"):
                    replacement_symbol = candidate
                    break

            # Fallback to sector ETF if no direct replacement
            if not replacement_symbol:
                etf = get_sector_etf_fallback(lot.symbol)
                if etf and etf not in existing_symbols and etf in prices and prices[etf].close > Decimal("0"):
                    replacement_symbol = etf

            # Skip TLH if no valid replacement found (to avoid cash drag)
            if not replacement_symbol:
                continue

            # Calculate replacement shares BEFORE executing trades
            replacement_price = prices[replacement_symbol].close
            # FIX 3: Use ROUND_HALF_UP instead of ROUND_DOWN to minimize cash drag
            shares_to_buy = (proceeds / replacement_price).quantize(
                Decimal("0.000001"), rounding=ROUND_HALF_UP
            )

            # Only proceed if we can buy meaningful shares
            if shares_to_buy <= Decimal("0"):
                continue

            # FIX 4: Create replacement lot FIRST (atomic swap)
            replacement_lot = TaxLot.create(
                symbol=replacement_symbol,
                shares=shares_to_buy,
                cost_basis=replacement_price,
                acquisition_date=current_date,
                portfolio_id=state.portfolio_id,
            )
            state.lots.append(replacement_lot)

            # Now remove the original lot (after replacement exists)
            state.lots.remove(lot)

            # Record TLH sell trade
            sell_trade = SimulationTrade.create(
                timestamp=timestamp,
                symbol=lot.symbol,
                side=TradeSide.SELL,
                shares=lot.shares,
                price=price,
                reason=TradeReason.TLH_SELL,
                lot_id=lot.lot_id,
                notes=f"TLH harvest, loss ${abs(loss_amount):.2f}",
            )
            state.trades.append(sell_trade)

            # Record replacement buy trade
            buy_trade = SimulationTrade.create(
                timestamp=timestamp,
                symbol=replacement_symbol,
                side=TradeSide.BUY,
                shares=shares_to_buy,
                price=replacement_price,
                reason=TradeReason.TLH_REPLACEMENT_BUY,
                notes=f"TLH replacement for {lot.symbol}",
            )
            state.trades.append(buy_trade)

            # Update cash (net of sell proceeds - buy cost)
            buy_cost = shares_to_buy * replacement_price
            state.cash += proceeds - buy_cost

            # Update tracking
            state.realized_pnl += loss_amount
            state.harvested_losses += abs(loss_amount)
            state.wash_sale_symbols[lot.symbol] = current_date

            # FIX 1: Mark BOTH symbols as traded today to prevent ping-pong
            traded_today.add(lot.symbol)
            traded_today.add(replacement_symbol)

            # Update existing_symbols since we now own the replacement
            existing_symbols.add(replacement_symbol)

            # Track limits
            tlh_trades_today += 1
            harvested_today += abs(loss_amount)

        return state

    def record_snapshot(
        self,
        state: SimulationState,
        prices: dict[str, PriceData],
        current_date: date,
    ) -> SimulationState:
        """
        Record daily portfolio snapshot.

        Args:
            state: Current simulation state
            prices: Current prices
            current_date: Current date

        Returns:
            Updated state with new snapshot
        """
        holdings_value = Decimal("0")
        unrealized_pnl = Decimal("0")

        for lot in state.lots:
            if lot.symbol in prices:
                current_price = prices[lot.symbol].close
                lot_value = lot.shares * current_price
                holdings_value += lot_value
                unrealized_pnl += lot.shares * (current_price - lot.cost_basis)

        total_value = holdings_value + state.cash

        # Calculate returns
        if state.snapshots:
            prev_value = state.snapshots[-1].total_value
            daily_return = (total_value - prev_value) / prev_value if prev_value > 0 else Decimal("0")
            initial_value = self.config.initial_cash
            cumulative_return = (total_value - initial_value) / initial_value
        else:
            daily_return = Decimal("0")
            cumulative_return = Decimal("0")

        # Calculate sector weights
        current_holdings = {
            lot.symbol: lot.shares * prices[lot.symbol].close
            for lot in state.lots
            if lot.symbol in prices
        }
        sector_weights = calculate_sector_weights(current_holdings)

        snapshot = DailySnapshot(
            date=current_date,
            total_value=total_value,
            cash=state.cash,
            holdings_value=holdings_value,
            num_positions=len(set(lot.symbol for lot in state.lots)),
            num_lots=len(state.lots),
            unrealized_pnl=unrealized_pnl,
            realized_pnl=state.realized_pnl,
            daily_return=daily_return,
            cumulative_return=cumulative_return,
            sector_weights=sector_weights,
        )
        state.snapshots.append(snapshot)

        return state

    def should_rebalance(self, current_date: date, last_rebalance: Optional[date]) -> bool:
        """
        Check if rebalancing should occur on this date.

        Args:
            current_date: Current date
            last_rebalance: Date of last rebalance (None if never)

        Returns:
            True if should rebalance
        """
        if last_rebalance is None:
            return True

        days_since = (current_date - last_rebalance).days

        if self.config.rebalance_freq == "daily":
            return days_since >= 1
        elif self.config.rebalance_freq == "weekly":
            return days_since >= 7
        elif self.config.rebalance_freq == "monthly":
            return days_since >= 28
        elif self.config.rebalance_freq == "quarterly":
            return days_since >= 90

        return days_since >= 7  # Default to weekly

    def process_dividends(
        self,
        state: SimulationState,
        prices: dict[str, PriceData],
        current_date: date,
    ) -> SimulationState:
        """
        Process dividend payments for the portfolio.
        
        Simulates quarterly dividends based on average yield estimate.
        Dividends are paid to cash (matching Fidelity pattern).
        
        Args:
            state: Current simulation state
            prices: Current prices
            current_date: Current date
            
        Returns:
            Updated state with dividend cash added
        """
        if not self.config.enable_dividend_simulation:
            return state
        
        # Simulate dividends on quarter-end months (Mar, Jun, Sep, Dec)
        # Most S&P 500 dividends cluster around these dates
        if current_date.month not in [3, 6, 9, 12]:
            return state
        
        # Pay on the 20th or next available trading day
        # To avoid paying multiple times in a month (the bug we found),
        # we check if a dividend trade has already been recorded this month
        
        # Check if already paid this month
        month_start = date(current_date.year, current_date.month, 1)
        already_paid = False
        for trade in reversed(state.trades):
            if trade.reason == TradeReason.DIVIDEND and trade.timestamp.date() >= month_start:
                already_paid = True
                break
        
        if already_paid:
            return state
            
        # Only process on or after the 20th
        if current_date.day < 20:
            return state
        
        # Calculate quarterly dividend (annual yield / 4)
        quarterly_yield = self.config.average_dividend_yield / Decimal("4")
        
        # Calculate total dividend across all holdings
        total_dividend = Decimal("0")
        portfolio_value = Decimal("0")
        
        for lot in state.lots:
            if lot.symbol in prices:
                lot_value = lot.shares * prices[lot.symbol].close
                dividend = lot_value * quarterly_yield
                total_dividend += dividend
                portfolio_value += lot_value
        
        if total_dividend > Decimal("0"):
            # Add dividend to cash
            state.cash += total_dividend
            
            # Record dividend "trade" for tracking
            # We use a special trade entry
            trade = SimulationTrade.create(
                timestamp=datetime.combine(current_date, datetime.min.time()),
                symbol="CASH",
                side=TradeSide.BUY, # Cash inflow
                shares=Decimal("0"),
                price=Decimal("1"),
                reason=TradeReason.DIVIDEND,
                notes=f"Quarterly Dividend ({quarterly_yield*100:.2f}%) on ${portfolio_value:,.2f}",
            )
            # Override value calculation for dividend
            trade.value = total_dividend
            state.trades.append(trade)
        
        return state

