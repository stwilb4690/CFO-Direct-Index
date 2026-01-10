"""
Fidelity transaction CSV import and comparison utility.

Parses Fidelity transaction history CSVs and enables comparison with
simulated backtest trades.

Fidelity CSV format:
- Trade Date: "Sep 3, 2025" format
- Security: Company name (e.g., "Johnson & Johnson")
- Type: Buy, Sell, Deposit, Dividend (Of Cash), etc.
- Value: Dollar amount with $ and commas
- Long-Term Capital Gain: Optional
- Short-Term Capital Gain: Optional
"""

import csv
import re
from dataclasses import dataclass
from datetime import datetime, date
from decimal import Decimal
from pathlib import Path
from typing import Optional

import pandas as pd


# Company name to ticker symbol mapping
# This maps the Fidelity security names to standard tickers
COMPANY_TO_TICKER = {
    # Mega-caps
    "Apple Inc": "AAPL",
    "Microsoft Corp": "MSFT",
    "Nvidia Corporation": "NVDA",
    "Amazon.Com Inc": "AMZN",
    "Alphabet Inc Cap STK": "GOOGL",  # Could be GOOG too
    "Meta Platforms Inc": "META",
    "Berkshire Hathaway": "BRK.B",
    "Tesla Inc Com": "TSLA",
    "Unitedhealth Group": "UNH",
    "Johnson &johnson Com": "JNJ",
    "Johnson & Johnson": "JNJ",
    
    # Large-caps (Financial)
    "JPMorgan Chase &co.": "JPM",
    "JPMorgan Chase & Co": "JPM",
    "Visa Inc": "V",
    "Mastercard": "MA",
    "Bank America Corp": "BAC",
    "Wells Fargo Co New": "WFC",
    "Goldman Sachs Group": "GS",
    "Morgan Stanley Com": "MS",
    "Citigroup Inc": "C",
    "American Express Co": "AXP",
    "Schwab Charles Corp": "SCHW",
    "BlackRock Inc Com": "BLK",
    "PNC Financial": "PNC",
    "Capital One": "COF",
    "Truist Finl Corp Com": "TFC",
    "US Bancorp": "USB",
    "State Street Corp": "STT",
    "Northern Trust Corp": "NTRS",
    "M &T Bank Corp Com": "MTB",
    "Fifth Third Bancorp": "FITB",
    "Keycorp Com": "KEY",
    "Huntington": "HBAN",
    "Citizens Finl Group": "CFG",
    "Regions Financial": "RF",
    "Synchrony Financial": "SYF",
    
    # Large-caps (Tech)
    "Broadcom Inc Com": "AVGO",
    "Adobe Inc Com": "ADBE",
    "Salesforce Inc Com": "CRM",
    "Oracle Corp": "ORCL",
    "Cisco Systems Inc": "CSCO",
    "Accenture PLC": "ACN",
    "Intel Corp Com": "INTC",
    "Qualcomm Inc": "QCOM",
    "Texas Instruments": "TXN",
    "International Bus": "IBM",
    "Advanced Micro": "AMD",
    "Intuit Inc": "INTU",
    "Servicenow Inc Com": "NOW",
    "Applied Materials": "AMAT",
    "Lam Research Corp": "LRCX",
    "Micron Technology": "MU",
    "Analog Devices Inc": "ADI",
    "Kla Corp Com New": "KLAC",
    "Synopsys Inc": "SNPS",
    "Cadence Design": "CDNS",
    "Autodesk Inc": "ADSK",
    "Workday Inc Com": "WDAY",
    "Palo Alto Networks": "PANW",
    "Crowdstrike HLDGS": "CRWD",
    "Fortinet Inc Com": "FTNT",
    "Snowflake Inc.": "SNOW",
    "Datadog Inc CL A Com": "DDOG",
    "Palantir": "PLTR",
    "Arista Networks Inc": "ANET",
    
    # Healthcare
    "Eli Lilly &co Com": "LLY",
    "Eli Lilly & Co": "LLY",
    "Pfizer Inc": "PFE",
    "Merck &co. Inc Com": "MRK",
    "Abbvie Inc Com": "ABBV",
    "Thermo Fisher": "TMO",
    "Abbott Laboratories": "ABT",
    "Danaher Corporation": "DHR",
    "Bristol-Myers Squibb": "BMY",
    "Amgen Inc": "AMGN",
    "Medtronic PLC": "MDT",
    "Elevance Health Inc": "ELV",
    "CVS Health": "CVS",
    "Gilead Sciences Inc": "GILD",
    "Intuitive Surgical": "ISRG",
    "Vertex": "VRTX",
    "Boston Scientific": "BSX",
    "Regeneron": "REGN",
    "Stryker Corporation": "SYK",
    "Humana Inc": "HUM",
    "Hca Healthcare Inc": "HCA",
    "Zoetis Inc": "ZTS",
    "The CIGNA Group Com": "CI",
    "Mckesson Corp": "MCK",
    "Becton Dickinson &co": "BDX",
    "Edwards Lifesciences": "EW",
    "Biogen Inc Com": "BIIB",
    
    # Consumer
    "Procter And Gamble": "PG",
    "Coca-Cola Co": "KO",
    "Pepsico Inc": "PEP",
    "Costco Wholesale": "COST",
    "WalMart Inc Com": "WMT",
    "Home Depot Inc": "HD",
    "Mcdonald S Corp": "MCD",
    "Nike Inc Class B Com": "NKE",
    "Starbucks Corp Com": "SBUX",
    "Lowes Companies Inc": "LOW",
    "TJX Companies Inc": "TJX",
    "Target Corp": "TGT",
    "Booking Holdings Inc": "BKNG",
    "Marriott": "MAR",
    "Hilton Worldwide": "HLT",
    "Chipotle Mexican": "CMG",
    "Colgate-Palmolive Co": "CL",
    "Mondelez Intl Inc": "MDLZ",
    "Kimberly-Clark Corp": "KMB",
    "General Mills Inc": "GIS",
    "Kraft Heinz Co Com": "KHC",
    "Yum! Brands Inc": "YUM",
    "Ross Stores Inc": "ROST",
    "Dollar Gen Corp New": "DG",
    "Kroger Co Com": "KR",
    
    # Industrial
    "Caterpillar Inc Com": "CAT",
    "Boeing Co": "BA",
    "Union Pac Corp Com": "UNP",
    "Honeywell": "HON",
    "Lockheed Martin Corp": "LMT",
    "RTX Corporation Com": "RTX",
    "Deere & Co": "DE",
    "GE Aerospace Com New": "GE",
    "General Dynamics": "GD",
    "Northrop Grumman": "NOC",
    "United Parcel": "UPS",
    "Fedex Corp Com": "FDX",
    "Illinois Tool Works": "ITW",
    "Emerson Electric Co": "EMR",
    "3M Co": "MMM",
    "CSX Corp Com Usd1": "CSX",
    "Norfolk Southn Corp": "NSC",
    "Parker-Hannifin Corp": "PH",
    "Eaton Corporation": "ETN",
    "Waste Management Inc": "WM",
    "Republic Services": "RSG",
    "Trane Technologies": "TT",
    "Cummins Inc": "CMI",
    "Paccar Inc Com": "PCAR",
    "L3harris": "LHX",
    "Transdigm Group Inc": "TDG",
    
    # Energy
    "Exxon Mobil Corp Com": "XOM",
    "Chevron Corp New Com": "CVX",
    "Conocophillips Com": "COP",
    "Schlumberger Limited": "SLB",
    "Eog Resources Inc": "EOG",
    "Marathon Petroleum": "MPC",
    "Phillips 66": "PSX",
    "Valero Energy Corp": "VLO",
    "Williams Cos Inc Com": "WMB",
    "Occidental Pete Corp": "OXY",
    "Kinder Morgan Inc": "KMI",
    "Devon Energy Corp": "DVN",
    "Halliburton Co Com": "HAL",
    "Baker Hughes Company": "BKR",
    "Coterra Energy Inc": "CTRA",
    "Diamondback Energy": "FANG",
    "Cheniere Energy Inc": "LNG",
    "Oneok Inc Com": "OKE",
    "Targa Resources Corp": "TRGP",
    
    # Utilities
    "Nextera Energy Inc": "NEE",
    "Duke Energy Corp New": "DUK",
    "Southern Co": "SO",
    "Dominion Energy Inc": "D",
    "American Elec PWR Co": "AEP",
    "Exelon Corp Com NPV": "EXC",
    "Sempra Com": "SRE",
    "Xcel Energy Inc Com": "XEL",
    "Wec Energy Group Inc": "WEC",
    "Consolidated Edison": "ED",
    "Public SVC": "PEG",
    "Eversource Energy": "ES",
    "Entergy Corp": "ETR",
    "Edison International": "EIX",
    "Dte Energy Co": "DTE",
    "Atmos Energy Corp": "ATO",
    "Centerpoint Energy": "CNP",
    "CMS Energy Corp Com": "CMS",
    "Alliant Energy Corp": "LNT",
    "Evergy Inc Com": "EVRG",
    "Ameren Corp Com": "AEE",
    "PPL Corp Com Usd0.01": "PPL",
    "Firstenergy Corp Com": "FE",
    "NRG Energy Inc": "NRG",
    "Constellation Energy": "CEG",
    "PG&E Corp Com NPV": "PCG",
    "Nisource Inc Com": "NI",
    
    # REITs
    "Prologis Inc. Com": "PLD",
    "American Tower Corp": "AMT",
    "Equinix Inc Com": "EQIX",
    "Crown Castle Inc Com": "CCI",
    "Public Storage Oper": "PSA",
    "Realty Income Corp": "O",
    "Digital Realty Trust": "DLR",
    "Simon Property Group": "SPG",
    "Welltower Inc Com": "WELL",
    "Extra Space Storage": "EXR",
    "Vici PPTYS Inc Com": "VICI",
    "Avalonbay": "AVB",
    "Equity Residential": "EQR",
    "Invitation Homes Inc": "INVH",
    "Ventas Inc": "VTR",
    "Iron MTN Inc Del Com": "IRM",
    "Sba Communications": "SBAC",
    
    # Communications
    "AT&T Inc Com Usd1": "T",
    "Verizon": "VZ",
    "T-Mobile US Inc Com": "TMUS",
    "Comcast Corp": "CMCSA",
    "Disney Walt Co Com": "DIS",
    "Netflix Inc": "NFLX",
    "Charter": "CHTR",
    "Warner Bros": "WBD",
    "Electronic Arts Inc": "EA",
    "Take-Two Interactive": "TTWO",
    "Activision Blizzard": "ATVI",
    "Live Nation": "LYV",
    "Fox Corp CL A Com": "FOXA",
    "Fox Corp CL B Com": "FOX",
    
    # Other notable companies
    "GE Vernova Inc Com": "GEV",
    "Vistra Corp Com": "VST",
    "Uber Technologies": "UBER",
    "Airbnb Inc Com CL A": "ABNB",
    "Doordash Inc CL A": "DASH",
    "Coinbase Global Inc": "COIN",
    "Robinhood MKTS Inc": "HOOD",
    "Microstrategy Com": "MSTR",
    "Roblox Corp CL A": "RBLX",
    "Axon Enterprise Inc": "AXON",
    "Applovin Corp Com CL": "APP",
    "Carvana Co CL A": "CVNA",
    "Block Inc CL A": "SQ",
    "Paypal HLDGS Inc Com": "PYPL",
    "Atlassian": "TEAM",
    "Cloudflare Inc CL A": "NET",
    
    # Additional companies from CSV
    "Snap-On Inc": "SNA",
    "Box Inc CL A": "BOX",
    "Iac Inc Com New": "IAC",
    "Insmed Inc": "INSM",
    "Henry Jack": "JKHY",
    "Amkor Technology I": "AMKR",
    "Omnicom Group Inc": "OMC",
    "Labcorp Holdings Inc": "LH",
    "Ralliant Corp Com": "RAIL", 
    "First Citizens": "FCNCA",
    "Axis Cap HLDGS LTD": "AXS",
    "Liberty Media Corp": "LSXMA",
    "Twilio Inc CL A": "TWLO",
    "Dover Corp Com": "DOV",
    "West Pharmaceutical": "WST",
    "Arthur J. Gallagher": "AJG",
    "Avalonbay": "AVB",
    "Hologic Inc": "HOLX",
    "Metlife Inc Com": "MET",
    "Church &dwight Co": "CHD",
    "Church & Dwight": "CHD",
    "Ulta Beauty Inc Com": "ULTA",
    "Biomarin": "BMRN",
    "Autozone Inc Com": "AZO",
    "Ford MTR Co Del Com": "F",
    "Wabtec Com": "WAB",
    "Marsh &mclennan": "MMC",
    "Marsh & McLennan": "MMC",
    "Eastman Chem Co Com": "EMN",
    "Booz Allen Hamilton": "BAH",
    "Western Digital": "WDC",
    "Las Vegas Sands Corp": "LVS",
    "Martin Marietta": "MLM",
    "Stanley Black &": "SWK",
    "Ares Management": "ARES",
    "Dexcom Inc": "DXCM",
    "Akamai Technologies": "AKAM",
    "Willis Towers Watson": "WTW",
    "Smurfit Westrock PLC": "SW",
    "Renaissancere HLDGS": "RNR",
    "Hubbell Inc Com": "HUBB",
    "Essex Property Trust": "ESS",
    "CRH Ord Eur 0.32": "CRH",
    "Texas Pacific Land": "TPL",
    "Kenvue Inc Com": "KVUE",
    "Fiserv Inc Com STK": "FI",
    "McCormick &company": "MKC",
    "McCormick & Company": "MKC",
    "Lennar Corp Com": "LEN",
    "Keysight": "KEYS",
    "Copart Inc Com": "CPRT",
    "Old Dominion Freight": "ODFL",
    "PTC Inc": "PTC",
    "LPL Financial": "LPLA",
    "Jabil Inc Com": "JBL",
    "Pulte Group Inc Com": "PHM",
    "eBay Inc. Com": "EBAY",
    "Weyerhaeuser Co MTN": "WY",
    "Lyondellbasell": "LYB",
    "Microchip Technology": "MCHP",
    "Moodys Corp Com": "MCO",
    "MSCI Inc": "MSCI",
    "Air Products And": "APD",
    "Hubspot Inc": "HUBS",
    "Lululemon Athletica": "LULU",
    "Waters Corp": "WAT",
    "Ferguson Enterprises": "FERG",
    "Williams-Sonoma Inc": "WSM",
    "Hunt J.B. Transport": "JBHT",
    "Ball Corp Com NPV": "BALL",
    "Aon PLC SHS CL A": "AON",
    "Arch Capital Group": "ACGL",
    "Sprouts Farmers": "SFM",
    "Johnson Controls": "JCI",
    "Super Micro Computer": "SMCI",
    "Monolithic Power": "MPWR",
    "Te Connectivity PLC": "TEL",
    "Xylem Inc Com": "XYL",
    "Blackstone Inc": "BX",
    "Expeditors": "EXPD",
    "Zebra Technologies": "ZBRA",
    "Garmin LTD Com": "GRMN",
    "Carrier Global": "CARR",
    "On Semiconductor": "ON",
    "Tractor Supply Co": "TSCO",
    "International": "IFF",
    "Interactive Brokers": "IBKR",
    "Broadridge Financial": "BR",
    "Fidelity National": "FIS",
    "Reliance Inc Com NPV": "RS",
    "Monster Beverage": "MNST",
    "Marvell Technology": "MRVL",
    "Ameriprise Financial": "AMP",
    "Leidos Holdings Inc": "LDOS",
    "Docusign Inc Com": "DOCU",
    "Draftkings Inc New": "DKNG",
    "Darden Restaurants": "DRI",
    "Idex Corp Com": "IEX",
    "Constellation Brands": "STZ",
    "Entegris Inc": "ENTG",
    "American": "AEE",  # Ameren
    "Eagle Matls Inc Com": "EXP",
    "Gartner Inc Com": "IT",
    "Principal Financial": "PFG",
    "Natera Inc Com": "NTRA",
    "Ralph Lauren Corp": "RL",
    "DuPont De Nemours": "DD",
    "CDW Corp Com Usd0.01": "CDW",
    "Fidelity Natl": "FIS",
    "General MTRS Co Com": "GM",
    "Pool Corp Com": "POOL",
    "Align Technology Inc": "ALGN",
    "Verisign Inc": "VRSN",
    "Fastenal Com STK": "FAST",
    "Fair Isaac Corp": "FICO",
    "Pinterest Inc CL A": "PINS",
    "Everest Group LTD": "EG",
    "Factset Research": "FDS",
    "NXP Semiconductors": "NXPI",
    "Tradeweb MKTS Inc CL": "TW",
    "EQT Corp Com": "EQT",
    "Ametek Inc Com": "AME",
    "Carlisle Companies": "CSL",
    "Hershey Company Com": "HSY",
    "Assurant Inc": "AIZ",
    "Intuit Inc": "INTU",
    "Nucor Corp Com": "NUE",
    "Newmont Corp Com": "NEM",
    "Best Buy Co Inc Com": "BBY",
    "Invitation Homes Inc": "INVH",
    "RB Global Inc Com": "RBA",
    "Ovintiv Inc Com": "OVV",
    "Dell Technologies": "DELL",
    "Illumina Inc Com": "ILMN",
    "Zoom Communications": "ZM",
    "American Water Works": "AWK",
    "Carnival Corp Com": "CCL",
    "Expedia Group Inc": "EXPE",
    "Sherwin-Williams Co": "SHW",
    "Quanta Services Com": "PWR",
    "Cintas Corp": "CTAS",
    "Steris PLC Ord": "STE",
    "Progressive Corp Com": "PGR",
    "Netapp Inc": "NTAP",
    "CBOE Global Markets": "CBOE",
    "Clorox Co Com": "CLX",
    "Royalty Pharma PLC": "RPRX",
    "Vertiv Holdings Co": "VRT",
    "Birkenstock Holding": "BIRK",
    "Dollar Tree Inc": "DLTR",
    "Automatic Data": "ADP",
    "Idexx Laboratories": "IDXX",
    "Equifax Inc Com": "EFX",
    "Alnylam": "ALNY",
    "Quest Diagnostics": "DGX",
    "Aptiv PLC Ord": "APTV",
    "Builders Firstsource": "BLDR",
    "Cognizant Technology": "CTSH",
    "International Paper": "IP",
    "Verisk Analytics Inc": "VRSK",
    "Estee Lauder": "EL",
    "Ingersoll Rand Inc": "IR",
    "Centene Corp": "CNC",
    "Textron Inc": "TXT",
    "Global Payments Inc": "GPN",
    "Expand Energy": "EXE",
    "Jacobs Solutions Inc": "J",
    "Baxter International": "BAX",
    "Packaging Corp Of": "PKG",
    "Alexandria Real": "ARE",
    "Columbia Sportswear": "COLM",
    "Veeva Systems Inc": "VEEV",
    "Rli Corp Com Usd1.00": "RLI",
    "Zimmer Biomet": "ZBH",
    "Nasdaq Inc": "NDAQ",
    "Tyson Foods Inc": "TSN",
    "Flutter": "FLUT",
    "Cardinal Health Inc": "CAH",
    "Carmax Inc": "KMX",
    "GE Healthcare": "GEHC",
    "Amphenol Corp Class": "APH",
    "Old Republic": "ORI",
    "Transunion Com": "TRU",
    "Travelers Companies": "TRV",
    "LKQ Corp": "LKQ",
    "Loews Corp Com": "L",
    "Unum Group": "UNM",
    "Mettler-Toledo": "MTD",
    "Grainger W W Inc Com": "GWW",
    "Oreilly Automotive": "ORLY",
    "DR Horton Inc Com": "DHI",
    "Iqvia HLDGS Inc Com": "IQV",
    "Sun Communities Inc": "SUI",
    "Brown & Brown Inc": "BRO",
    "United Airls HLDGS": "UAL",
    "Southwest Airlines": "LUV",
    "Delta Air Lines Inc": "DAL",
    "Royal Caribbean": "RCL",
    "United Rentals Inc": "URI",
    "Roper Technologies": "ROP",
    "Howmet Aerospace Inc": "HWM",
    "Vulcan Materials Co": "VMC",
    "Genuine Parts Co Com": "GPC",
    "Veralto Corp Com SHS": "VLTO",
    "Molina Healthcare": "MOH",
    "Amentum Holdings Inc": "AMTM",
    "Tyler Technologies": "TYL",
    "Agilent Technologies": "A",
    "Avantor Inc Com": "AVTR",
    "Gen Digital Inc Com": "GEN",
    "Godaddy Inc CL A": "GDDY",
    "HP Inc Com": "HPQ",
    "Viatris Inc Com": "VTRS",
    "Heico Corp New CL A": "HEI.A",
    "Heico Corp New Com": "HEI",
    "Teledyne": "TDY",
    "Corpay Inc Com SHS": "CPAY",
    "The Trade Desk Inc": "TTD",
    "Corteva Inc Com": "CTVA",
    "Watsco Inc": "WSO",
    "Healthpeak": "DOC",
    "First Solar Inc": "FSLR",
    "Cme Group Inc Com": "CME",
    "Cencora Inc Com": "COR",
    "Motorola Solutions": "MSI",
    "Bank Of New York": "BK",
    "Aflac Inc Com": "AFL",
    "Berkley W R Corp Com": "WRB",
    "Ameren Corp Com": "AEE",
    "Gaming & Leisure P": "GLPI",
    "Sysco Corp": "SYY",
    "Teradyne Inc Com": "TER",
    "Cincinnati Financial": "CINF",
    "Linde PLC Com": "LIN",
    "Insulet Corp": "PODD",
    "WP Carey Inc Com": "WPC",
    "Costar Group Inc": "CSGP",
    "Mongodb Inc CL A": "MDB",
    "Mid-Amer Apt CMNTYS": "MAA",
    "Okta Inc CL A": "OKTA",
    "Cna Financial Corp": "CNA",
    "Resmed Inc": "RMD",
    "Smucker J M Co Com": "SJM",
    "Udr Inc": "UDR",
    "Masco Corp Com": "MAS",
    "Cbre Group Inc Com": "CBRE",
    "SS&C Technologies": "SSNC",
    "Otis Worldwide Corp": "OTIS",
    "Conagra Brands Inc": "CAG",
    "Fortive Corp Com": "FTV",
    "Hewlett Packard": "HPE",
    "Avery Dennison Corp": "AVY",
    "Zscaler Inc Com": "ZS",
    "Apollo Global MGMT": "APO",
    "Corning Inc": "GLW",
    "Price T Rowe Groups": "TROW",
    "American Financial": "AFG",
    "CF Industries": "CF",
    "Raymond James Finl": "RJF",
    "Assured Guaranty LTD": "AGO",
    "Revvity Inc Com": "RVTY",
    "F5 Inc Com": "FFIV",
    "Seagate Technology": "STX",
    "Cooper Cos Inc Com": "COO",
    "PVH Corporation Com": "PVH",
    "Prudential Financial": "PRU",
    "Paychex Inc Com": "PAYX",
    "Allstate Corp Com": "ALL",
    "Markel Group Inc Com": "MKL",
    "The Hartford": "HIG",
    "Trimble Inc Com": "TRMB",
    "Archer-Daniels-Midla": "ADM",
    "Host Hotels &resorts": "HST",
}



@dataclass
class FidelityTransaction:
    """Parsed Fidelity transaction."""
    trade_date: date
    security_name: str
    ticker: Optional[str]  # Mapped ticker or None if unknown
    transaction_type: str  # Buy, Sell, Deposit, Dividend, etc.
    value: Decimal
    long_term_gain: Optional[Decimal]
    short_term_gain: Optional[Decimal]
    
    @property
    def is_buy(self) -> bool:
        return self.transaction_type.lower() == "buy"
    
    @property
    def is_sell(self) -> bool:
        return self.transaction_type.lower() == "sell"
    
    @property
    def is_dividend(self) -> bool:
        return "dividend" in self.transaction_type.lower()
    
    @property
    def is_deposit(self) -> bool:
        return self.transaction_type.lower() == "deposit"


def parse_date(date_str: str) -> date:
    """Parse Fidelity date format (e.g., 'Sep 3, 2025')."""
    # Remove quotes if present
    date_str = date_str.strip().strip('"')
    try:
        return datetime.strptime(date_str, "%b %d, %Y").date()
    except ValueError:
        # Try alternate format
        return datetime.strptime(date_str, "%B %d, %Y").date()


def parse_value(value_str: str) -> Decimal:
    """Parse Fidelity value format (e.g., '$3,500,000.00' or '-$1,234.56')."""
    if not value_str or value_str.strip() == "":
        return Decimal("0")
    
    # Remove quotes, $, and commas
    cleaned = value_str.strip().strip('"').replace("$", "").replace(",", "")
    
    # Handle negative values
    if cleaned.startswith("-"):
        return Decimal(cleaned)
    elif cleaned.startswith("(") and cleaned.endswith(")"):
        # Accounting format for negatives
        return -Decimal(cleaned[1:-1])
    
    return Decimal(cleaned) if cleaned else Decimal("0")


def map_company_to_ticker(company_name: str) -> Optional[str]:
    """Map a Fidelity company name to a ticker symbol."""
    # Exact match first
    if company_name in COMPANY_TO_TICKER:
        return COMPANY_TO_TICKER[company_name]
    
    # Try case-insensitive match
    company_lower = company_name.lower().strip()
    for key, ticker in COMPANY_TO_TICKER.items():
        if key.lower() == company_lower:
            return ticker
    
    # Try partial match (starts with)
    for key, ticker in COMPANY_TO_TICKER.items():
        if company_lower.startswith(key.lower()[:10]):
            return ticker
    
    return None


def parse_fidelity_csv(filepath: str | Path) -> list[FidelityTransaction]:
    """
    Parse a Fidelity transaction history CSV file.
    
    Args:
        filepath: Path to the CSV file
        
    Returns:
        List of FidelityTransaction objects
    """
    transactions = []
    filepath = Path(filepath)
    
    with open(filepath, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        
        for row in reader:
            trade_date_str = row.get("Trade Date", "").strip()
            security = row.get("Security", "").strip()
            tx_type = row.get("Type", "").strip()
            value_str = row.get("Value", "")
            lt_gain_str = row.get("Long-Term Capital Gain", "")
            st_gain_str = row.get("Short-Term Capital Gain", "")
            
            # Skip header rows or empty rows
            if not trade_date_str or trade_date_str == "Trade Date":
                continue
            
            # Skip "Open Position" rows
            if tx_type.lower() == "open position":
                continue
                
            try:
                trade_date = parse_date(trade_date_str)
            except ValueError:
                continue
            
            value = parse_value(value_str)
            lt_gain = parse_value(lt_gain_str) if lt_gain_str else None
            st_gain = parse_value(st_gain_str) if st_gain_str else None
            
            # Map company to ticker
            ticker = map_company_to_ticker(security)
            
            transactions.append(FidelityTransaction(
                trade_date=trade_date,
                security_name=security,
                ticker=ticker,
                transaction_type=tx_type,
                value=value,
                long_term_gain=lt_gain,
                short_term_gain=st_gain,
            ))
    
    return transactions


def summarize_transactions(transactions: list[FidelityTransaction]) -> dict:
    """
    Generate summary statistics from Fidelity transactions.
    
    Returns:
        Dictionary with summary metrics
    """
    if not transactions:
        return {"error": "No transactions found"}
    
    buys = [t for t in transactions if t.is_buy]
    sells = [t for t in transactions if t.is_sell]
    dividends = [t for t in transactions if t.is_dividend]
    deposits = [t for t in transactions if t.is_deposit]
    
    # Calculate totals
    total_buys = sum(abs(t.value) for t in buys)
    total_sells = sum(abs(t.value) for t in sells)
    total_dividends = sum(abs(t.value) for t in dividends)
    total_deposits = sum(abs(t.value) for t in deposits)
    
    # Count unique securities
    unique_securities = set(t.security_name for t in transactions if t.security_name)
    mapped_tickers = set(t.ticker for t in transactions if t.ticker)
    unmapped = [t.security_name for t in transactions if not t.ticker and t.security_name]
    
    # Date range
    dates = [t.trade_date for t in transactions]
    
    return {
        "date_range": {
            "start": min(dates).isoformat(),
            "end": max(dates).isoformat(),
        },
        "total_transactions": len(transactions),
        "buys": {
            "count": len(buys),
            "total_value": float(total_buys),
        },
        "sells": {
            "count": len(sells),
            "total_value": float(total_sells),
        },
        "dividends": {
            "count": len(dividends),
            "total_value": float(total_dividends),
        },
        "deposits": {
            "count": len(deposits),
            "total_value": float(total_deposits),
        },
        "unique_securities": len(unique_securities),
        "mapped_tickers": len(mapped_tickers),
        "unmapped_securities": len(set(unmapped)),
        "unmapped_list": list(set(unmapped))[:20],  # First 20 unmapped
    }


def to_dataframe(transactions: list[FidelityTransaction]) -> pd.DataFrame:
    """Convert transactions to a pandas DataFrame."""
    records = []
    for t in transactions:
        records.append({
            "trade_date": t.trade_date,
            "security_name": t.security_name,
            "ticker": t.ticker,
            "type": t.transaction_type,
            "value": float(t.value),
            "is_buy": t.is_buy,
            "is_sell": t.is_sell,
            "is_dividend": t.is_dividend,
            "is_deposit": t.is_deposit,
        })
    return pd.DataFrame(records)


def compare_with_backtest(
    fidelity_transactions: list[FidelityTransaction],
    backtest_trades_df: pd.DataFrame,
) -> dict:
    """
    Compare Fidelity transactions with simulated backtest trades.
    
    Args:
        fidelity_transactions: Parsed Fidelity transactions
        backtest_trades_df: DataFrame from BacktestResult.trades_to_dataframe()
        
    Returns:
        Comparison report dictionary
    """
    # Filter to just buys and sells
    fidelity_buys = [t for t in fidelity_transactions if t.is_buy]
    fidelity_sells = [t for t in fidelity_transactions if t.is_sell]
    
    backtest_buys = backtest_trades_df[backtest_trades_df["side"] == "BUY"]
    backtest_sells = backtest_trades_df[backtest_trades_df["side"] == "SELL"]
    
    # Summary comparison
    return {
        "fidelity": {
            "buy_count": len(fidelity_buys),
            "sell_count": len(fidelity_sells),
            "total_buy_value": float(sum(abs(t.value) for t in fidelity_buys)),
            "total_sell_value": float(sum(abs(t.value) for t in fidelity_sells)),
        },
        "backtest": {
            "buy_count": len(backtest_buys),
            "sell_count": len(backtest_sells),
            "total_buy_value": float(backtest_buys["value"].sum()) if len(backtest_buys) > 0 else 0,
            "total_sell_value": float(backtest_sells["value"].sum()) if len(backtest_sells) > 0 else 0,
        },
        "differences": {
            "buy_count_diff": len(fidelity_buys) - len(backtest_buys),
            "sell_count_diff": len(fidelity_sells) - len(backtest_sells),
        },
    }


if __name__ == "__main__":
    # Test with sample file
    import sys
    
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        # Default test path
        filepath = "transactions (4).csv"
    
    print(f"Parsing: {filepath}")
    transactions = parse_fidelity_csv(filepath)
    summary = summarize_transactions(transactions)
    
    print("\n=== Fidelity Transaction Summary ===")
    print(f"Date Range: {summary['date_range']['start']} to {summary['date_range']['end']}")
    print(f"Total Transactions: {summary['total_transactions']}")
    print(f"\nBuys: {summary['buys']['count']} (${summary['buys']['total_value']:,.2f})")
    print(f"Sells: {summary['sells']['count']} (${summary['sells']['total_value']:,.2f})")
    print(f"Dividends: {summary['dividends']['count']} (${summary['dividends']['total_value']:,.2f})")
    print(f"Deposits: {summary['deposits']['count']} (${summary['deposits']['total_value']:,.2f})")
    print(f"\nUnique Securities: {summary['unique_securities']}")
    print(f"Mapped to Tickers: {summary['mapped_tickers']}")
    print(f"Unmapped: {summary['unmapped_securities']}")
    
    if summary['unmapped_list']:
        print("\nFirst 20 unmapped securities:")
        for name in summary['unmapped_list']:
            print(f"  - {name}")
