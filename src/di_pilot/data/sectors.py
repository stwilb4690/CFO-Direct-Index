"""
Sector tracking for portfolio analysis.

Provides GICS sector mappings and functions to calculate sector weights
for portfolio vs benchmark comparison.
"""

from dataclasses import dataclass
from decimal import Decimal
from typing import Optional


# GICS Sector mappings - symbol to sector name
# Based on S&P 500 current classifications
SECTOR_MAPPINGS = {
    # Information Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology", 
    "AVGO": "Technology", "CRM": "Technology", "ORCL": "Technology",
    "AMD": "Technology", "ADBE": "Technology", "INTC": "Technology",
    "CSCO": "Technology", "IBM": "Technology", "TXN": "Technology",
    "QCOM": "Technology", "INTU": "Technology", "AMAT": "Technology",
    "NOW": "Technology", "MU": "Technology", "LRCX": "Technology",
    "ADI": "Technology", "KLAC": "Technology", "PANW": "Technology",
    "CRWD": "Technology", "FTNT": "Technology", "HPE": "Technology",
    "HPQ": "Technology", "DELL": "Technology", "SNPS": "Technology",
    "CDNS": "Technology", "MCHP": "Technology", "ON": "Technology",
    "NXPI": "Technology", "MPWR": "Technology", "KEYS": "Technology",
    "ANSS": "Technology", "FSLR": "Technology", "ENPH": "Technology",
    
    # Healthcare
    "UNH": "Healthcare", "JNJ": "Healthcare", "LLY": "Healthcare",
    "PFE": "Healthcare", "MRK": "Healthcare", "ABBV": "Healthcare",
    "TMO": "Healthcare", "ABT": "Healthcare", "DHR": "Healthcare",
    "BMY": "Healthcare", "AMGN": "Healthcare", "MDT": "Healthcare",
    "CI": "Healthcare", "ELV": "Healthcare", "CVS": "Healthcare",
    "GILD": "Healthcare", "ISRG": "Healthcare", "VRTX": "Healthcare",
    "BSX": "Healthcare", "REGN": "Healthcare", "SYK": "Healthcare",
    "HUM": "Healthcare", "ZBH": "Healthcare", "BIIB": "Healthcare",
    "MRNA": "Healthcare", "IDXX": "Healthcare", "IQV": "Healthcare",
    "DXCM": "Healthcare", "MTD": "Healthcare", "WAT": "Healthcare",
    "A": "Healthcare", "ALGN": "Healthcare", "HOLX": "Healthcare",
    
    # Financials
    "JPM": "Financials", "BAC": "Financials", "WFC": "Financials",
    "GS": "Financials", "MS": "Financials", "BRK.B": "Financials",
    "C": "Financials", "SCHW": "Financials", "USB": "Financials",
    "PNC": "Financials", "TFC": "Financials", "BLK": "Financials",
    "AXP": "Financials", "CB": "Financials", "MET": "Financials",
    "PRU": "Financials", "AIG": "Financials", "TRV": "Financials",
    "ALL": "Financials", "PGR": "Financials", "AFL": "Financials",
    "COF": "Financials", "DFS": "Financials", "BX": "Financials",
    "KKR": "Financials", "ICE": "Financials", "CME": "Financials",
    "SPGI": "Financials", "MCO": "Financials", "MSCI": "Financials",
    
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary", "TGT": "Consumer Discretionary",
    "TJX": "Consumer Discretionary", "CMG": "Consumer Discretionary",
    "MAR": "Consumer Discretionary", "HLT": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary", "GM": "Consumer Discretionary",
    "F": "Consumer Discretionary", "YUM": "Consumer Discretionary",
    "LULU": "Consumer Discretionary", "BBY": "Consumer Discretionary",
    "ABNB": "Consumer Discretionary", "EBAY": "Consumer Discretionary",
    "DPZ": "Consumer Discretionary", "ROST": "Consumer Discretionary",
    "ORLY": "Consumer Discretionary", "AZO": "Consumer Discretionary",
    "DHI": "Consumer Discretionary", "LEN": "Consumer Discretionary",
    
    # Communication Services
    "GOOGL": "Communication Services", "GOOG": "Communication Services",
    "META": "Communication Services", "NFLX": "Communication Services",
    "DIS": "Communication Services", "VZ": "Communication Services",
    "T": "Communication Services", "CMCSA": "Communication Services",
    "TMUS": "Communication Services", "WBD": "Communication Services",
    "PARA": "Communication Services", "FOX": "Communication Services",
    "FOXA": "Communication Services", "EA": "Communication Services",
    "TTWO": "Communication Services", "MTCH": "Communication Services",
    "CHTR": "Communication Services", "OMC": "Communication Services",
    "IPG": "Communication Services", "LYV": "Communication Services",
    
    # Consumer Staples
    "PG": "Consumer Staples", "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "COST": "Consumer Staples", "WMT": "Consumer Staples", "PM": "Consumer Staples",
    "MO": "Consumer Staples", "MDLZ": "Consumer Staples", "CL": "Consumer Staples",
    "KMB": "Consumer Staples", "GIS": "Consumer Staples", "K": "Consumer Staples",
    "KR": "Consumer Staples", "SYY": "Consumer Staples", "HSY": "Consumer Staples",
    "KDP": "Consumer Staples", "CHD": "Consumer Staples", "CPB": "Consumer Staples",
    "STZ": "Consumer Staples", "EL": "Consumer Staples", "ADM": "Consumer Staples",
    "HRL": "Consumer Staples", "MKC": "Consumer Staples", "SJM": "Consumer Staples",
    
    # Energy
    "XOM": "Energy", "CVX": "Energy", "COP": "Energy", "SLB": "Energy",
    "EOG": "Energy", "MPC": "Energy", "PXD": "Energy", "PSX": "Energy",
    "VLO": "Energy", "OXY": "Energy", "DVN": "Energy", "HAL": "Energy",
    "KMI": "Energy", "WMB": "Energy", "BKR": "Energy", "FANG": "Energy",
    "HES": "Energy", "CTRA": "Energy", "MRO": "Energy", "APA": "Energy",
    "OKE": "Energy", "TRGP": "Energy",
    
    # Industrials
    "UNP": "Industrials", "HON": "Industrials", "UPS": "Industrials",
    "CAT": "Industrials", "BA": "Industrials", "GE": "Industrials",
    "RTX": "Industrials", "LMT": "Industrials", "DE": "Industrials",
    "MMM": "Industrials", "FDX": "Industrials", "CSX": "Industrials",
    "NSC": "Industrials", "EMR": "Industrials", "ITW": "Industrials",
    "NOC": "Industrials", "GD": "Industrials", "DAL": "Industrials",
    "UAL": "Industrials", "LUV": "Industrials", "CMI": "Industrials",
    "PCAR": "Industrials", "ETN": "Industrials", "JCI": "Industrials",
    "WM": "Industrials", "RSG": "Industrials", "CTAS": "Industrials",
    "ROK": "Industrials", "CARR": "Industrials", "OTIS": "Industrials",
    
    # Utilities
    "NEE": "Utilities", "DUK": "Utilities", "SO": "Utilities",
    "D": "Utilities", "AEP": "Utilities", "EXC": "Utilities",
    "SRE": "Utilities", "XEL": "Utilities", "PEG": "Utilities",
    "WEC": "Utilities", "ES": "Utilities", "ED": "Utilities",
    "DTE": "Utilities", "EIX": "Utilities", "PCG": "Utilities",
    "ETR": "Utilities", "FE": "Utilities", "AWK": "Utilities",
    "AEE": "Utilities", "CMS": "Utilities", "CNP": "Utilities",
    "PPL": "Utilities", "NI": "Utilities", "EVRG": "Utilities",
    
    # Materials
    "LIN": "Materials", "APD": "Materials", "SHW": "Materials",
    "FCX": "Materials", "NEM": "Materials", "NUE": "Materials",
    "ECL": "Materials", "DD": "Materials", "DOW": "Materials",
    "PPG": "Materials", "VMC": "Materials", "MLM": "Materials",
    "CTVA": "Materials", "ALB": "Materials", "IFF": "Materials",
    "CE": "Materials", "EMN": "Materials", "BALL": "Materials",
    "PKG": "Materials", "IP": "Materials", "AVY": "Materials",
    "CF": "Materials", "MOS": "Materials", "FMC": "Materials",
    
    # Real Estate
    "PLD": "Real Estate", "AMT": "Real Estate", "EQIX": "Real Estate",
    "SPG": "Real Estate", "O": "Real Estate", "WELL": "Real Estate",
    "CCI": "Real Estate", "DLR": "Real Estate", "AVB": "Real Estate",
    "EQR": "Real Estate", "VICI": "Real Estate", "VTR": "Real Estate",
    "SBAC": "Real Estate", "ESS": "Real Estate", "MAA": "Real Estate",
    "UDR": "Real Estate", "PEAK": "Real Estate", "ARE": "Real Estate",
    "KIM": "Real Estate", "REG": "Real Estate", "HST": "Real Estate",
    "BXP": "Real Estate", "CPT": "Real Estate", "IRM": "Real Estate",
}

# Standard GICS sector list
GICS_SECTORS = [
    "Technology",
    "Healthcare", 
    "Financials",
    "Consumer Discretionary",
    "Communication Services",
    "Consumer Staples",
    "Energy",
    "Industrials",
    "Utilities",
    "Materials",
    "Real Estate",
]


def get_sector(symbol: str) -> str:
    """Get GICS sector for a symbol. Returns 'Other' if unknown."""
    return SECTOR_MAPPINGS.get(symbol.upper(), "Other")


@dataclass
class SectorWeights:
    """Portfolio sector weights at a point in time."""
    date: str
    weights: dict[str, float]  # sector -> weight
    
    def drift_from(self, benchmark: "SectorWeights") -> dict[str, float]:
        """Calculate drift from benchmark weights."""
        drift = {}
        for sector in GICS_SECTORS:
            port_weight = self.weights.get(sector, 0.0)
            bench_weight = benchmark.weights.get(sector, 0.0)
            drift[sector] = port_weight - bench_weight
        return drift


def calculate_sector_weights(
    holdings: dict[str, Decimal],  # symbol -> market value
) -> dict[str, float]:
    """
    Calculate sector weights from portfolio holdings.
    
    Args:
        holdings: Dictionary of symbol to market value
        
    Returns:
        Dictionary of sector to weight (0-1)
    """
    total_value = sum(holdings.values())
    if total_value == 0:
        return {}
    
    sector_values: dict[str, Decimal] = {}
    for symbol, value in holdings.items():
        sector = get_sector(symbol)
        sector_values[sector] = sector_values.get(sector, Decimal("0")) + value
    
    return {
        sector: float(value / total_value)
        for sector, value in sector_values.items()
    }


def calculate_sector_drift(
    portfolio_weights: dict[str, float],
    benchmark_weights: dict[str, float],
) -> dict[str, float]:
    """
    Calculate sector drift between portfolio and benchmark.
    
    Returns:
        Dictionary of sector to drift (positive = overweight)
    """
    drift = {}
    all_sectors = set(portfolio_weights.keys()) | set(benchmark_weights.keys())
    
    for sector in all_sectors:
        port = portfolio_weights.get(sector, 0.0)
        bench = benchmark_weights.get(sector, 0.0)
        drift[sector] = port - bench
    
    return drift


def format_sector_drift_table(drift: dict[str, float]) -> str:
    """Format sector drift as ASCII table for reports."""
    lines = []
    lines.append("| Sector | Drift |")
    lines.append("|--------|-------|")
    
    for sector in GICS_SECTORS:
        d = drift.get(sector, 0.0)
        sign = "+" if d >= 0 else ""
        lines.append(f"| {sector} | {sign}{d:.2%} |")
    
    return "\n".join(lines)
