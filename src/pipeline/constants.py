"""
Module with project constants

This module is developed as part of a Master's thesis entitled
"Application of Survival Models to a Real Sample of Medical Data".

The thesis is carried out at the Institute of Artificial Inteligence,
Faculty of Electrical Engineering and Informatics,
Technical University of Košice, during the academic year 2025/2026.

The research is based on medical data provided by
the Louis Pasteur University Hospital in Košice.
"""

import os
from pathlib import Path

# DATA PATH constants
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_ROOT: Path = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "DATA")).resolve()
IMPUTED_CACHE_DIR: Path = DATA_ROOT / "imputed_cache"
PIVOTED_CACHE_DIR: Path = DATA_ROOT / "pivoted_cache"
OUTPUT_ROOT: Path = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "output")).resolve()

ORIGINAL_WAVE_1: Path = DATA_ROOT / "original" / "wave_1.xlsx"
ORIGINAL_WAVE_2: Path = DATA_ROOT / "original" / "wave_2.xlsx"
ORIGINAL_WAVE_3: Path = DATA_ROOT / "original" / "wave_3.xlsx"
ORIGINAL_WAVE_4: Path = DATA_ROOT / "original" / "wave_4.xlsx"

PREPARED_WAVE_1: Path = DATA_ROOT / "prepared" / "wave_1.xlsx"
PREPARED_WAVE_2: Path = DATA_ROOT / "prepared" / "wave_2.xlsx"
PREPARED_WAVE_3: Path = DATA_ROOT / "prepared" / "wave_3.xlsx"
PREPARED_WAVE_4: Path = DATA_ROOT / "prepared" / "wave_4.xlsx"


# DATA constants
EVENT_COL: str = "Závažnosť priebehu ochorenia"
DURATION_COL: str = "duration"
ADMISSION_COL: str = "Dátum príjmu"
DISCHARGE_COL: str = "Dátum prepustenia"
TIME_COL: str = "time"
GROUP_COL: str = "ID"
GENDER_COL: str = "Pohlavie"
AGE_COL: str = "Vek"

# ORIGINAL DATASET
ORIGINAL_RENAME_MAP: dict[str, str] = {"Isoprinosine, ": "Isoprinosine"}
ORIGINAL_MEASUREMENT_COLS: list[str] = [
    "S-Bil-T",
    "S-AST",
    "S-ALT",
    "S-GMT",
    "S-ALP",
    "S-CB",
    "S-Na",
    "S-K",
    "S-CL",
    "S-CRP",
    "S-Alb",
    "S-Gluk",
    "S-Urea",
    "S-Kreat",
    "S-CK",
    "S-CK-MB",
    "P-Laktát",
    "S-FER",
    "HGB",
    "WBC",
    "PLT",
    "Neu abs",
    "Eo abs",
    "Ly abs",
    "PT (INR)",
    "APTT-R",
    "Fib",
    "CD3+",
    "CD4+",
    "CD8+",
    "CD4+/CD8+",
    "PDW",
    "S-PBNP",
    "S-IL6",
    "CD19+",
    "NK",
    "S-VITD",
    "NE/LY(NLR)",
    "D-dimér HS",
]
ORIGINAL_DROP_COLS: list[str] = [
    "HLN Dg.",
    "Diagnózy",
    "DRG výkony",
    "Liečba",
    "SVLZ správy",
    "Mikrobiológia ",
    "Epikríza",
    "Terajšie ochorenie",
    "Dôvod hospitalizácie",
    "Objektívny nález",
    "Osobná anamnéza",
    "Lieková anamnéza",
    "Návyková anamnéza",
    "Epidemiologická anamnéza",
    "Meno",
    "Kód príjmu",
    "Kód prepustenia",
    "S-IgA",
    "S-Chol",
    "S-Ig M",
    "S-IgG",
    "S-AMS",
    "S-LD",
    "S-KM",
    "S-Bil-D",
]
# PREPARED DATASET
PREPARED_RENAME_MAP: dict[str, str] = {
    "Isoprinosine, ": "Isoprinosine",
    "24814 | CALCIFEROL BBP 7,5 MG/ML": "24814 | CALCIFEROL BBP 7.5 MG/ML",
    "93105 DEGAN ": "93105 DEGAN",
    "24949 CODEIN ": "24949 CODEIN",
}
PREPARED_COLS_TO_USE: list[str] = [
    "A04.7",
    "Závažnosť priebehu ochorenia",
    "Fajčenie",
    "Alkohol",
    "SatO2 %",
    "Hypertenzia",
    "Diabetes mellitus",
    "Kardiovaskulárne ochorenia",
    "Chronické respiračné ochorenia",
    "Renálne ochorenia",
    "Pečeňové ochorenia",
    "Onkologické ochorenia",
    "Imunosupresia",
    "Vakcinácia",
    "Typ vakcíny",
    "Počet dávok",
    "MD652 | FABIFLU TABLETS",
    "MD656 IV-BECT 6MG (ivermectin)",
    "5042D | VEKLURY",
    "9547D | PAXLOVID",
    "LAGEVRIO",
    "00584 | PYRIDOXIN LÉČIVA INJ",
    "24836 | ACIDUM ASCORBICUM BBP",
    "24814 | CALCIFEROL BBP 7.5 MG/ML",
    "00498 | MAGNESIUM SULFURICUM BBP 100 MG/ML INJEKČNÝ ROZTOK",
    "00449 | EREVIT 300 MG/ML",
    "89145 | VITAMIN C-INJEKTOPAS",
    "92973 ALPHA D3",
    "02963 | PREDNISON 20 LÉČIVA",
    "00269 | PREDNISON 5 LÉČIVA",
    "84090 | DEXAMED 6",
    "1275C | DEXAMETAZÓN KRKA",
    "MD661 BIODEXONE-DEXAMETHASONE",
    "2410B HYDROCORTISONE",
    "3242C | OLUMIANT 4 MG",
    "Anakinra",
    "RoActemra",
    "34045 | POLYOXIDONIUM 6 MG",
    "87299 | IMUNOR",
    "56930 IMMODIN",
    "Isoprinosine",
    "3879d INOMED",
    "35715 Azithromycin",
    "45954 Ceftriaxon",
    "0471B MOLOXIN",
    "9819A MOXIFLOXACIN",
    "58730 CIPROFLOXACIN KABI 200",
    "58746 CIPROFLOXACINKABI 400",
    "05044 OZZION",
    "4147C OMEMYL",
    "89662 NOLPAZA",
    "39397 PANTOPRAZOL",
    "62916 SMECTA",
    "30639 REASEC",
    "84370 LAGOSA",
    "93105 DEGAN",
    "94918 AMBROBENE",
    "24859 PENTOXYPHILLINUM",
    "8893 ACC INJEKT",
    "24949 CODEIN",
    "26846 OXANTIL",
    "FRAXIPARIN",
    "CLEXANE",
    "FRAGMIN",
    "ASPIRIN",
    "ANOPYRIN",
    "Prekonal COVID-19",
]
# PARSING
PARSING_STATIC_COLS: list[str] = [
    "Dátum príjmu",
    "Dátum prepustenia",
    "Závažnosť priebehu ochorenia",
    "Pohlavie",
    "Vek",
    "A04.7",
    "Fajčenie",
    "Alkohol",
    "SatO2 %",
    "Hypertenzia",
    "Diabetes mellitus",
    "Kardiovaskulárne ochorenia",
    "Chronické respiračné ochorenia",
    "Renálne ochorenia",
    "Pečeňové ochorenia",
    "Onkologické ochorenia",
    "Imunosupresia",
    "Vakcinácia",
    "Počet dávok",
    "MD652 | FABIFLU TABLETS",
    "MD656 IV-BECT 6MG (ivermectin)",
    "5042D | VEKLURY",
    "9547D | PAXLOVID",
    "LAGEVRIO",
    "00584 | PYRIDOXIN LÉČIVA INJ",
    "24836 | ACIDUM ASCORBICUM BBP",
    "24814 | CALCIFEROL BBP 7.5 MG/ML",
    "00498 | MAGNESIUM SULFURICUM BBP 100 MG/ML INJEKČNÝ ROZTOK",
    "00449 | EREVIT 300 MG/ML",
    "89145 | VITAMIN C-INJEKTOPAS",
    "92973 ALPHA D3",
    "02963 | PREDNISON 20 LÉČIVA",
    "00269 | PREDNISON 5 LÉČIVA",
    "84090 | DEXAMED 6",
    "1275C | DEXAMETAZÓN KRKA",
    "MD661 BIODEXONE-DEXAMETHASONE",
    "2410B HYDROCORTISONE",
    "3242C | OLUMIANT 4 MG",
    "Anakinra",
    "RoActemra",
    "34045 | POLYOXIDONIUM 6 MG",
    "87299 | IMUNOR",
    "56930 IMMODIN",
    "Isoprinosine",
    "3879d INOMED",
    "35715 Azithromycin",
    "45954 Ceftriaxon",
    "0471B MOLOXIN",
    "9819A MOXIFLOXACIN",
    "58730 CIPROFLOXACIN KABI 200",
    "58746 CIPROFLOXACINKABI 400",
    "05044 OZZION",
    "4147C OMEMYL",
    "89662 NOLPAZA",
    "39397 PANTOPRAZOL",
    "62916 SMECTA",
    "30639 REASEC",
    "84370 LAGOSA",
    "93105 DEGAN",
    "94918 AMBROBENE",
    "24859 PENTOXYPHILLINUM",
    "8893 ACC INJEKT",
    "24949 CODEIN",
    "26846 OXANTIL",
    "FRAXIPARIN",
    "CLEXANE",
    "FRAGMIN",
    "ASPIRIN",
    "ANOPYRIN",
    "Prekonal COVID-19",
]
# PIVOTING
PIVOT_STATIC_COLS: list[str] = [
    "Dátum príjmu",
    "Dátum prepustenia",
    "Závažnosť priebehu ochorenia",
    "Pohlavie",
    "Vek",
]
