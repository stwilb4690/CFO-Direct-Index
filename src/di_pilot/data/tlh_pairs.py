"""
TLH replacement pairs for wash sale compliance.

When harvesting a loss in Stock A, buy correlated Stock B from same sector.
This maintains market exposure while realizing the tax loss.

The IRS considers securities "substantially identical" if they are:
- Same company (obviously)
- Convertible securities of the same company
- Options on the same security

Different companies in the same sector are NOT substantially identical,
so PEP -> KO is a valid TLH swap.
"""

# GICS Sector-based replacement pairs
# Each stock maps to 2-3 correlated alternatives in same industry
TLH_REPLACEMENT_PAIRS = {
    # =========================================================================
    # INFORMATION TECHNOLOGY
    # =========================================================================

    # Software - Enterprise
    "MSFT": ["CRM", "ORCL", "NOW"],
    "CRM": ["MSFT", "NOW", "WDAY"],
    "ORCL": ["MSFT", "SAP", "IBM"],
    "NOW": ["CRM", "WDAY", "MSFT"],
    "ADBE": ["CRM", "INTU", "NOW"],
    "INTU": ["ADBE", "CRM", "WDAY"],
    "IBM": ["ORCL", "CSCO", "ACN"],
    "ACN": ["IBM", "CTSH", "INFY"],

    # Software - Security
    "PANW": ["CRWD", "FTNT", "ZS"],
    "CRWD": ["PANW", "FTNT", "ZS"],
    "FTNT": ["PANW", "CRWD", "ZS"],

    # Semiconductors
    "NVDA": ["AMD", "AVGO", "QCOM"],
    "AMD": ["NVDA", "INTC", "QCOM"],
    "AVGO": ["NVDA", "QCOM", "TXN"],
    "INTC": ["AMD", "TXN", "QCOM"],
    "QCOM": ["AVGO", "NVDA", "TXN"],
    "TXN": ["AVGO", "INTC", "ADI"],
    "ADI": ["TXN", "MCHP", "ON"],
    "MU": ["NVDA", "AMD", "INTC"],
    "LRCX": ["AMAT", "KLAC", "ASML"],
    "AMAT": ["LRCX", "KLAC", "ASML"],
    "KLAC": ["LRCX", "AMAT", "ASML"],

    # Hardware / Consumer Electronics
    "AAPL": ["DELL", "HPQ", "CSCO"],
    "DELL": ["HPQ", "AAPL", "CSCO"],
    "HPQ": ["DELL", "HPE", "CSCO"],
    "HPE": ["DELL", "HPQ", "CSCO"],
    "CSCO": ["JNPR", "ANET", "HPE"],

    # IT Services
    "V": ["MA", "PYPL", "AXP"],
    "MA": ["V", "PYPL", "AXP"],
    "PYPL": ["V", "MA", "SQ"],

    # =========================================================================
    # COMMUNICATION SERVICES
    # =========================================================================

    # Interactive Media
    "GOOGL": ["META", "NFLX", "DIS"],
    "GOOG": ["META", "NFLX", "DIS"],
    "META": ["GOOGL", "SNAP", "PINS"],

    # Entertainment / Streaming
    "NFLX": ["DIS", "WBD", "PARA"],
    "DIS": ["NFLX", "WBD", "CMCSA"],
    "WBD": ["DIS", "PARA", "NFLX"],
    "PARA": ["WBD", "DIS", "FOX"],

    # Telecom
    "T": ["VZ", "TMUS", "CMCSA"],
    "VZ": ["T", "TMUS", "CMCSA"],
    "TMUS": ["T", "VZ", "CMCSA"],

    # =========================================================================
    # CONSUMER DISCRETIONARY
    # =========================================================================

    # E-commerce / Retail
    "AMZN": ["WMT", "TGT", "COST"],
    "HD": ["LOW", "TGT", "COST"],
    "LOW": ["HD", "TGT", "BBY"],
    "TGT": ["WMT", "COST", "HD"],
    "BBY": ["LOW", "HD", "TGT"],
    "EBAY": ["AMZN", "ETSY", "W"],

    # Automotive
    "TSLA": ["GM", "F", "RIVN"],
    "GM": ["F", "TSLA", "STLA"],
    "F": ["GM", "TSLA", "STLA"],

    # Restaurants
    "MCD": ["SBUX", "YUM", "CMG"],
    "SBUX": ["MCD", "CMG", "DPZ"],
    "CMG": ["MCD", "SBUX", "YUM"],
    "YUM": ["MCD", "CMG", "DPZ"],

    # Apparel / Footwear
    "NKE": ["LULU", "UAA", "VFC"],
    "LULU": ["NKE", "UAA", "GPS"],

    # Hotels / Leisure
    "MAR": ["HLT", "H", "WH"],
    "HLT": ["MAR", "H", "WH"],
    "BKNG": ["ABNB", "EXPE", "MAR"],
    "ABNB": ["BKNG", "EXPE", "MAR"],

    # =========================================================================
    # CONSUMER STAPLES
    # =========================================================================

    # Beverages
    "KO": ["PEP", "MNST", "KDP"],
    "PEP": ["KO", "MNST", "KDP"],
    "MNST": ["KO", "PEP", "KDP"],
    "KDP": ["KO", "PEP", "MNST"],

    # Food
    "MDLZ": ["HSY", "K", "GIS"],
    "HSY": ["MDLZ", "K", "GIS"],
    "GIS": ["K", "CPB", "SJM"],
    "K": ["GIS", "CPB", "MDLZ"],
    "CPB": ["GIS", "K", "SJM"],

    # Household / Personal Care
    "PG": ["CL", "KMB", "CHD"],
    "CL": ["PG", "KMB", "CHD"],
    "KMB": ["PG", "CL", "CHD"],
    "CHD": ["PG", "CL", "KMB"],

    # Retail - Staples
    "COST": ["WMT", "TGT", "KR"],
    "WMT": ["COST", "TGT", "KR"],
    "KR": ["WMT", "COST", "SYY"],

    # Tobacco
    "PM": ["MO", "BTI", "LVS"],
    "MO": ["PM", "BTI", "RAI"],

    # =========================================================================
    # FINANCIALS
    # =========================================================================

    # Banks - Large
    "JPM": ["BAC", "WFC", "C"],
    "BAC": ["JPM", "WFC", "USB"],
    "WFC": ["JPM", "BAC", "PNC"],
    "C": ["JPM", "BAC", "GS"],
    "USB": ["PNC", "TFC", "BAC"],
    "PNC": ["USB", "TFC", "WFC"],
    "TFC": ["USB", "PNC", "FITB"],

    # Investment Banks
    "GS": ["MS", "JPM", "C"],
    "MS": ["GS", "JPM", "SCHW"],
    "SCHW": ["MS", "IBKR", "ETFC"],

    # Insurance
    "BRK.B": ["MET", "PRU", "AIG"],
    "MET": ["PRU", "AIG", "AFL"],
    "PRU": ["MET", "AIG", "LNC"],
    "AIG": ["MET", "PRU", "TRV"],
    "TRV": ["CB", "AIG", "ALL"],
    "CB": ["TRV", "ALL", "PGR"],
    "ALL": ["PGR", "TRV", "CB"],
    "PGR": ["ALL", "TRV", "GEICO"],
    "AFL": ["MET", "PRU", "UNM"],

    # Asset Management
    "BLK": ["BX", "KKR", "APO"],
    "BX": ["BLK", "KKR", "APO"],
    "KKR": ["BX", "APO", "BLK"],

    # Credit Cards / Payment
    "AXP": ["DFS", "COF", "V"],
    "DFS": ["AXP", "COF", "SYF"],
    "COF": ["AXP", "DFS", "SYF"],

    # =========================================================================
    # HEALTHCARE
    # =========================================================================

    # Pharma - Large
    "JNJ": ["PFE", "MRK", "ABBV"],
    "PFE": ["MRK", "JNJ", "BMY"],
    "MRK": ["PFE", "LLY", "ABBV"],
    "LLY": ["NVO", "MRK", "BMY"],
    "ABBV": ["BMY", "AMGN", "GILD"],
    "BMY": ["ABBV", "PFE", "MRK"],
    "NVO": ["LLY", "SNY", "AZN"],

    # Biotech
    "AMGN": ["GILD", "REGN", "BIIB"],
    "GILD": ["AMGN", "BIIB", "VRTX"],
    "REGN": ["AMGN", "VRTX", "BIIB"],
    "BIIB": ["AMGN", "GILD", "REGN"],
    "VRTX": ["REGN", "GILD", "MRNA"],
    "MRNA": ["BNTX", "PFE", "VRTX"],

    # Health Insurance / Managed Care
    "UNH": ["CVS", "CI", "ELV"],
    "CVS": ["UNH", "CI", "WBA"],
    "CI": ["UNH", "ELV", "HUM"],
    "ELV": ["UNH", "CI", "HUM"],
    "HUM": ["CI", "ELV", "CNC"],

    # Medical Devices
    "ABT": ["MDT", "BSX", "SYK"],
    "MDT": ["ABT", "BSX", "SYK"],
    "BSX": ["ABT", "MDT", "EW"],
    "SYK": ["ABT", "ZBH", "MDT"],
    "ISRG": ["INTU", "SYK", "ABT"],
    "DHR": ["TMO", "A", "PKI"],
    "TMO": ["DHR", "A", "PKI"],

    # =========================================================================
    # ENERGY
    # =========================================================================

    # Integrated Oil
    "XOM": ["CVX", "COP", "EOG"],
    "CVX": ["XOM", "COP", "SLB"],
    "COP": ["XOM", "CVX", "EOG"],

    # E&P
    "EOG": ["PXD", "DVN", "OXY"],
    "PXD": ["EOG", "DVN", "FANG"],
    "DVN": ["EOG", "PXD", "OXY"],
    "OXY": ["DVN", "EOG", "COP"],

    # Services
    "SLB": ["HAL", "BKR", "CVX"],
    "HAL": ["SLB", "BKR", "NOV"],
    "BKR": ["SLB", "HAL", "NOV"],

    # Midstream
    "KMI": ["WMB", "ET", "EPD"],
    "WMB": ["KMI", "ET", "EPD"],

    # =========================================================================
    # INDUSTRIALS
    # =========================================================================

    # Aerospace / Defense
    "BA": ["LMT", "RTX", "GD"],
    "LMT": ["RTX", "BA", "NOC"],
    "RTX": ["LMT", "BA", "GD"],
    "GD": ["LMT", "RTX", "NOC"],
    "NOC": ["LMT", "GD", "RTX"],

    # Airlines
    "DAL": ["UAL", "LUV", "AAL"],
    "UAL": ["DAL", "LUV", "AAL"],
    "LUV": ["DAL", "UAL", "AAL"],
    "AAL": ["DAL", "UAL", "LUV"],

    # Railroads
    "UNP": ["CSX", "NSC", "CP"],
    "CSX": ["UNP", "NSC", "CP"],
    "NSC": ["UNP", "CSX", "CP"],

    # Industrial Conglomerates
    "HON": ["MMM", "EMR", "ITW"],
    "MMM": ["HON", "ITW", "EMR"],
    "GE": ["HON", "MMM", "RTX"],
    "EMR": ["ROK", "HON", "ETN"],

    # Machinery
    "CAT": ["DE", "CMI", "PCAR"],
    "DE": ["CAT", "AGCO", "CNH"],
    "CMI": ["CAT", "PCAR", "OSK"],

    # Logistics / Delivery
    "UPS": ["FDX", "XPO", "JBHT"],
    "FDX": ["UPS", "XPO", "EXPD"],

    # =========================================================================
    # MATERIALS
    # =========================================================================

    # Chemicals
    "LIN": ["APD", "SHW", "ECL"],
    "APD": ["LIN", "ECL", "DD"],
    "SHW": ["PPG", "LIN", "ECL"],
    "DD": ["DOW", "LYB", "EMN"],
    "DOW": ["LYB", "DD", "CE"],

    # Mining / Metals
    "FCX": ["NEM", "GOLD", "SCCO"],
    "NEM": ["GOLD", "FCX", "AEM"],
    "NUE": ["STLD", "CLF", "X"],

    # =========================================================================
    # UTILITIES
    # =========================================================================

    "NEE": ["DUK", "SO", "D"],
    "DUK": ["NEE", "SO", "AEP"],
    "SO": ["NEE", "DUK", "D"],
    "D": ["NEE", "SO", "EXC"],
    "AEP": ["DUK", "XEL", "EIX"],
    "EXC": ["D", "SRE", "PEG"],
    "SRE": ["EXC", "PCG", "EIX"],
    "XEL": ["AEP", "WEC", "DTE"],

    # =========================================================================
    # REAL ESTATE
    # =========================================================================

    # REITs - Industrial / Data Centers
    "PLD": ["AMT", "EQIX", "DLR"],
    "AMT": ["CCI", "EQIX", "PLD"],
    "EQIX": ["DLR", "AMT", "PLD"],
    "CCI": ["AMT", "SBAC", "EQIX"],
    "DLR": ["EQIX", "AMT", "PLD"],

    # REITs - Retail
    "SPG": ["O", "VICI", "REG"],
    "O": ["SPG", "NNN", "VICI"],

    # REITs - Residential
    "AVB": ["EQR", "ESS", "MAA"],
    "EQR": ["AVB", "ESS", "UDR"],

    # REITs - Healthcare
    "WELL": ["VTR", "PEAK", "OHI"],
    "VTR": ["WELL", "PEAK", "OHI"],
}


def get_replacement_candidates(symbol: str, excluded_symbols: set) -> list[str]:
    """
    Get replacement candidates for TLH, excluding wash-sale restricted symbols.

    Args:
        symbol: Stock being sold
        excluded_symbols: Set of symbols that cannot be bought (wash sale restricted)

    Returns:
        List of valid replacement candidates, ordered by preference
    """
    candidates = TLH_REPLACEMENT_PAIRS.get(symbol.upper(), [])
    return [c for c in candidates if c.upper() not in {s.upper() for s in excluded_symbols}]


def get_sector_etf_fallback(symbol: str) -> str | None:
    """
    Fallback: If no direct replacement available, use sector ETF.

    This is less ideal (different risk profile) but maintains market exposure.
    """
    # Map symbols to sector ETFs based on GICS sector
    SECTOR_ETF_MAP = {
        # Technology
        "XLK": [
            "AAPL", "MSFT", "NVDA", "AVGO", "CRM", "ORCL", "AMD", "ADBE", "INTC",
            "CSCO", "IBM", "TXN", "QCOM", "INTU", "AMAT", "NOW", "MU", "LRCX",
            "ADI", "KLAC", "PANW", "CRWD", "FTNT", "HPE", "HPQ", "DELL",
        ],
        # Healthcare
        "XLV": [
            "UNH", "JNJ", "LLY", "PFE", "MRK", "ABBV", "TMO", "ABT", "DHR",
            "BMY", "AMGN", "MDT", "CI", "ELV", "CVS", "GILD", "ISRG", "VRTX",
            "BSX", "REGN", "SYK", "HUM", "ZBH", "BIIB", "MRNA", "NVO",
        ],
        # Financials
        "XLF": [
            "JPM", "BAC", "WFC", "GS", "MS", "BRK.B", "C", "SCHW", "USB",
            "PNC", "TFC", "BLK", "AXP", "CB", "MET", "PRU", "AIG", "TRV",
            "ALL", "PGR", "AFL", "COF", "DFS", "BX", "KKR",
        ],
        # Consumer Discretionary
        "XLY": [
            "AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TGT", "TJX",
            "CMG", "MAR", "HLT", "BKNG", "GM", "F", "YUM", "LULU", "BBY",
            "ABNB", "EBAY", "DPZ", "W",
        ],
        # Communication Services
        "XLC": [
            "GOOGL", "GOOG", "META", "NFLX", "DIS", "VZ", "T", "CMCSA",
            "TMUS", "WBD", "PARA", "FOX", "FOXA", "EA", "TTWO", "MTCH",
        ],
        # Consumer Staples
        "XLP": [
            "PG", "KO", "PEP", "COST", "WMT", "PM", "MO", "MDLZ", "CL",
            "KMB", "GIS", "K", "KR", "SYY", "HSY", "KDP", "CHD", "CPB",
        ],
        # Energy
        "XLE": [
            "XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PXD", "PSX", "VLO",
            "OXY", "DVN", "HAL", "KMI", "WMB", "BKR", "FANG",
        ],
        # Industrials
        "XLI": [
            "UNP", "HON", "UPS", "CAT", "BA", "GE", "RTX", "LMT", "DE",
            "MMM", "FDX", "CSX", "NSC", "EMR", "ITW", "NOC", "GD", "DAL",
            "UAL", "LUV", "CMI", "PCAR",
        ],
        # Utilities
        "XLU": [
            "NEE", "DUK", "SO", "D", "AEP", "EXC", "SRE", "XEL", "PEG",
            "WEC", "ES", "ED", "DTE", "EIX", "PCG", "ETR", "FE",
        ],
        # Materials
        "XLB": [
            "LIN", "APD", "SHW", "FCX", "NEM", "NUE", "ECL", "DD", "DOW",
            "PPG", "VMC", "MLM", "CTVA", "ALB", "IFF", "CE", "EMN",
        ],
        # Real Estate
        "XLRE": [
            "PLD", "AMT", "EQIX", "SPG", "O", "WELL", "CCI", "DLR", "AVB",
            "EQR", "VICI", "VTR", "SBAC", "ESS", "MAA", "UDR", "PEAK",
        ],
    }

    symbol_upper = symbol.upper()
    for etf, symbols in SECTOR_ETF_MAP.items():
        if symbol_upper in [s.upper() for s in symbols]:
            return etf
    return None


def has_replacement_available(symbol: str) -> bool:
    """Check if a symbol has TLH replacement candidates defined."""
    return symbol.upper() in TLH_REPLACEMENT_PAIRS


import pandas as pd

def get_dynamic_replacement_candidate(
    loss_symbol: str, 
    sector_tickers: list[str], 
    price_history: pd.DataFrame, 
    excluded_symbols: set
) -> str | None:
    """
    Find a replacement using real-time correlation (Professional Standard).
    
    Args:
        loss_symbol: The stock we are harvesting loss on.
        sector_tickers: List of other tickers in the same sector.
        price_history: DataFrame of historical closes (columns=tickers, index=date).
        excluded_symbols: Wash sale restricted symbols.
        
    Returns:
        Symbol of the most correlated eligible stock, or None.
    """
    if loss_symbol not in price_history.columns:
        return None
        
    # calculate correlation with target stock
    # We only care about the last 60 days for recent correlation
    recent_prices = price_history.tail(60)
    correlations = recent_prices[sector_tickers].corrwith(recent_prices[loss_symbol])
    
    # Sort by highest correlation
    candidates = correlations.sort_values(ascending=False)
    
    for symbol, corr in candidates.items():
        # Must be highly correlated (>0.85) but not identical (1.0)
        if symbol != loss_symbol and corr > 0.85 and corr < 0.99:
            if symbol not in excluded_symbols:
                return symbol
                
    return None
