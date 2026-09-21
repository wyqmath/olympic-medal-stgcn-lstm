import os
from collections import OrderedDict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
GEN_DIR = os.path.join(HERE, "gen")
PANEL_OUT = os.path.join(GEN_DIR, "panel_clean.csv")
REPORT_OUT = os.path.join(HERE, "panel_build_report.md")

PANEL_COLS = [
    "country", "ioc", "year",
    "gold", "silver", "bronze", "total",
    "host_flag",
    "gdp_total_2011usd_million", "gdp_per_capita_2011usd", "population",
    "gdp_gapfill",
]
GAPFILL_VALUES = ("raw", "proxy_deu", "proxy_sum", "proxy_hist", "carry2y", "missing")
GAPFILL_PROXIES = ("proxy_deu", "proxy_sum", "proxy_hist")

GDP_PROXY_RULES = OrderedDict([
    ("East Germany", ("proxy_deu", ["Germany"])),
    ("West Germany", ("proxy_deu", ["Germany"])),
    ("United Team of Germany", ("proxy_deu", ["Germany"])),
    ("Unified Team", ("proxy_sum", [
        "Armenia", "Azerbaijan", "Belarus", "Georgia", "Kazakhstan", "Kyrgyzstan",
        "Moldova", "Russia", "Tajikistan", "Turkmenistan", "Ukraine", "Uzbekistan",
    ])),
    ("Serbia and Montenegro", ("proxy_sum", ["Serbia", "Montenegro"])),
    ("FR Yugoslavia", ("proxy_sum", ["Serbia", "Montenegro"])),
    ("Australasia", ("proxy_sum", ["Australia", "New Zealand"])),
    ("Bohemia", ("proxy_hist", ["Czech Republic", "Czechoslovakia"])),
    ("Russian Empire", ("proxy_hist", ["Russia"])),
])
CARRY_MIN_YEAR = 2023
CARRY_FROM_YEAR = 2022

IOC_MAP = {
    'Afghanistan': 'AFG',
    'Albania': 'ALB',
    'Algeria': 'ALG',
    'Argentina': 'ARG',
    'Armenia': 'ARM',
    'Australasia': 'ANZ',
    'Australia': 'AUS',
    'Austria': 'AUT',
    'Azerbaijan': 'AZE',
    'Bahamas': 'BAH',
    'Bahrain': 'BRN',
    'Barbados': 'BAR',
    'Belarus': 'BLR',
    'Belgium': 'BEL',
    'Bermuda': 'BER',
    'Bohemia': 'BOH',
    'Botswana': 'BOT',
    'Brazil': 'BRA',
    'British West Indies': 'BWI',
    'Bulgaria': 'BUL',
    'Burkina Faso': 'BUR',
    'Burundi': 'BDI',
    'Cabo Verde': 'CPV',
    'Cameroon': 'CMR',
    'Canada': 'CAN',
    'Ceylon': 'SRI',
    'Chile': 'CHI',
    'China': 'CHN',
    'Chinese Taipei': 'TPE',
    'Colombia': 'COL',
    'Costa Rica': 'CRC',
    'Croatia': 'CRO',
    'Cuba': 'CUB',
    'Cyprus': 'CYP',
    'Czech Republic': 'CZE',
    'Czechoslovakia': 'TCH',
    'Denmark': 'DEN',
    'Djibouti': 'DJI',
    'Dominica': 'DMA',
    'Dominican Republic': 'DOM',
    'East Germany': 'GDR',
    'Ecuador': 'ECU',
    'Egypt': 'EGY',
    'Eritrea': 'ERI',
    'Estonia': 'EST',
    'Ethiopia': 'ETH',
    'FR Yugoslavia': 'YUG',
    'Fiji': 'FIJ',
    'Finland': 'FIN',
    'Formosa': 'TPE',
    'France': 'FRA',
    'Gabon': 'GAB',
    'Georgia': 'GEO',
    'Germany': 'GER',
    'Ghana': 'GHA',
    'Great Britain': 'GBR',
    'Greece': 'GRE',
    'Grenada': 'GRN',
    'Guatemala': 'GUA',
    'Guyana': 'GUY',
    'Haiti': 'HAI',
    'Hong Kong': 'HKG',
    'Hungary': 'HUN',
    'Iceland': 'ISL',
    'Independent Olympic Athletes': 'IOA',
    'Independent Olympic Participants': 'IOP',
    'India': 'IND',
    'Indonesia': 'INA',
    'Iran': 'IRI',
    'Iraq': 'IRQ',
    'Ireland': 'IRL',
    'Israel': 'ISR',
    'Italy': 'ITA',
    'Ivory Coast': 'CIV',
    'Jamaica': 'JAM',
    'Japan': 'JPN',
    'Jordan': 'JOR',
    'Kazakhstan': 'KAZ',
    'Kenya': 'KEN',
    'Kosovo': 'KOS',
    'Kuwait': 'KUW',
    'Kyrgyzstan': 'KGZ',
    'Latvia': 'LAT',
    'Lebanon': 'LIB',
    'Lithuania': 'LTU',
    'Luxembourg': 'LUX',
    'Macedonia': 'MKD',
    'Malaysia': 'MAS',
    'Mauritius': 'MRI',
    'Mexico': 'MEX',
    'Mixed team': '',
    'Moldova': 'MDA',
    'Mongolia': 'MGL',
    'Montenegro': 'MNE',
    'Morocco': 'MAR',
    'Mozambique': 'MOZ',
    'Namibia': 'NAM',
    'Netherlands': 'NED',
    'Netherlands Antilles': 'AHO',
    'New Zealand': 'NZL',
    'Niger': 'NIG',
    'Nigeria': 'NGR',
    'North Korea': 'PRK',
    'North Macedonia': 'MKD',
    'Norway': 'NOR',
    'Pakistan': 'PAK',
    'Panama': 'PAN',
    'Paraguay': 'PAR',
    'Peru': 'PER',
    'Philippines': 'PHI',
    'Poland': 'POL',
    'Portugal': 'POR',
    'Puerto Rico': 'PUR',
    'Qatar': 'QAT',
    'ROC': 'ROC',
    'Refugee Olympic Team': 'EOR',
    'Romania': 'ROU',
    'Russia': 'RUS',
    'Russian Empire': 'RUS',
    'Saint Lucia': 'LCA',
    'Samoa': 'SAM',
    'San Marino': 'SMR',
    'Saudi Arabia': 'KSA',
    'Senegal': 'SEN',
    'Serbia': 'SRB',
    'Serbia and Montenegro': 'SCG',
    'Singapore': 'SGP',
    'Slovakia': 'SVK',
    'Slovenia': 'SLO',
    'South Africa': 'RSA',
    'South Korea': 'KOR',
    'Soviet Union': 'URS',
    'Spain': 'ESP',
    'Sri Lanka': 'SRI',
    'Sudan': 'SUD',
    'Suriname': 'SUR',
    'Sweden': 'SWE',
    'Switzerland': 'SUI',
    'Syria': 'SYR',
    'Taiwan': 'TPE',
    'Tajikistan': 'TJK',
    'Tanzania': 'TAN',
    'Thailand': 'THA',
    'Togo': 'TOG',
    'Tonga': 'TGA',
    'Trinidad and Tobago': 'TTO',
    'Tunisia': 'TUN',
    'Turkey': 'TUR',
    'Turkmenistan': 'TKM',
    'Uganda': 'UGA',
    'Ukraine': 'UKR',
    'Unified Team': 'EUN',
    'United Arab Emirates': 'UAE',
    'United States': 'USA',
    'United Team of Germany': 'EUA',
    'Uruguay': 'URU',
    'Uzbekistan': 'UZB',
    'Venezuela': 'VEN',
    'Vietnam': 'VIE',
    'Virgin Islands': 'ISV',
    'West Germany': 'FRG',
    'Yugoslavia': 'YUG',
    'Zambia': 'ZAM',
    'Zimbabwe': 'ZIM',
}

MANUAL_OVERRIDES = {
    'Formosa': ('TPE', 'USA',
                'athletes has a single Team=Formosa row miscoded as USA; 1960 Formosa is the Chinese Taipei delegation -> TPE'),
    'Taiwan': ('TPE', '<absent>',
               '1968 Taiwan is the Chinese Taipei delegation, same NOC as Chinese Taipei -> TPE'),
    'British West Indies': ('BWI', '<absent>',
                            'historical IOC code BWI (1960 West Indies Federation team); athletes files it under Team=West Indies Federation with the non-standard provisional code WIF'),
    'Ceylon': ('SRI', '<absent>',
               'Ceylon is the predecessor of Sri Lanka; athletes already uses SRI throughout (historical code CEY), keep the same code'),
    'FR Yugoslavia': ('YUG', '<absent>',
                      'historical IOC code YUG; athletes files the 1996/2000 team under SCG, colliding with the 2004 Serbia and Montenegro code, so keep the historical IOC code'),
    'Independent Olympic Athletes': ('IOA', '<absent>',
                                     'IOA is the official IOC code for the 2012/2016 Independent Olympic Athletes; athletes records them as Team=Individual Olympic Athletes with NOC IOA'),
    'Independent Olympic Participants': ('IOP', '<absent>',
                                         'historical IOC code IOP (1992 Independent Olympic Participants); athletes folds them into IOA, keep the historical code to separate them from 2016'),
    'ROC': ('ROC', '<absent>',
            'IOC code ROC (2020 Russian Olympic Committee); athletes also uses NOC=ROC but files the rows under Team=Russia (561 rows, 2020)'),
    'Russian Empire': ('RUS', '<absent>',
                       'historical IOC code RUS (1908/1912 Russian Empire), same code as modern Russia'),
    'United Team of Germany': ('EUA', '<absent>',
                               'historical IOC code EUA (1956/1960/1964 joint German team)'),
    'Ireland': ('IRL', 'IRL',
                'majority vote IRL (1603); GBR (6) are pre-1922 rows of Irish athletes competing for Great Britain, not the Ireland delegation'),
    'Lebanon': ('LIB', 'LIB',
                'majority vote LIB (236); LBN (17) is a variant miscoding of the same NOC, official IOC code is LIB'),
    'Russia': ('RUS', 'RUS',
               'majority vote RUS (3528); ROC (561) belongs to the 2020 delegation, which the medal table lists separately and which must not be merged into Russia'),
    'Mixed team': ('', '<absent>',
                   'multi-nation teams of 1896/1900/1904, not an NOC; leave the IOC code empty but keep the row'),
}

OLD_CHAIN_FILES = [
    ("combined_data.csv", "RawName"),
    ("combined_data_merged.csv", "country"),
    ("combined_data_updated.csv", "country"),
]


def die(msg):
    raise AssertionError(msg)


def log(msg=""):
    print(msg, flush=True)


def load_medals():
    path = os.path.join(DATA_DIR, "summerOly_medal_counts.csv")
    raw = pd.read_csv(path)
    log("[load] %s: %d rows, %d games"
        % (os.path.basename(path), len(raw), raw["Year"].nunique()))
    raw = raw.copy()
    raw["NOC_RAW"] = raw["NOC"]
    raw["country"] = raw["NOC"].str.replace(r"\?$", "", regex=True)
    raw["had_qmark"] = raw["NOC"].str.endswith("?")

    n_q = int(raw["had_qmark"].sum())
    q_labels = sorted(raw.loc[raw["had_qmark"], "NOC_RAW"].unique())
    q_years = sorted(int(y) for y in raw.loc[raw["had_qmark"], "Year"].unique())
    log("[load] '?' suffix: %d rows / %d labels, years %s" % (n_q, len(q_labels), q_years))

    if raw.duplicated(["country", "Year"]).any():
        dup = raw[raw.duplicated(["country", "Year"], keep=False)].sort_values(["Year", "country"])
        die("duplicate (country, year) rows after stripping '?', merging is not allowed:\n%s" % dup)
    if raw["country"].isna().any() or (raw["country"].str.strip() == "").any():
        die("empty country name after stripping '?'")

    for c in ["Gold", "Silver", "Bronze", "Total"]:
        if not pd.api.types.is_integer_dtype(raw[c]):
            die("%s is not an integer column: %s" % (c, raw[c].dtype))
    bad = raw[raw["Total"] != raw["Gold"] + raw["Silver"] + raw["Bronze"]]
    if len(bad):
        die("%d rows with Total != gold+silver+bronze:\n%s" % (len(bad), bad))

    for y in q_years:
        sub = raw[raw["Year"] == y]
        if sub["NOC_RAW"].nunique() != len(sub):
            die("edition %d has several rows for one label, the '?' handling needs a re-think" % y)
    return raw


def build_athletes_vote():
    a = pd.read_csv(os.path.join(DATA_DIR, "summerOly_athletes.csv"), usecols=["Team", "NOC"])
    v = a.groupby(["Team", "NOC"]).size().rename("n").reset_index().sort_values("n", ascending=False)
    votes = OrderedDict()
    for team, grp in v.groupby("Team", sort=False):
        votes[team] = list(zip(grp["NOC"], grp["n"]))
    return votes


def verify_ioc_map(labels, votes):
    missing = [l for l in labels if l not in IOC_MAP]
    extra = [k for k in IOC_MAP if k not in labels]
    if missing:
        die("IOC_MAP is missing %d medal-table labels: %s" % (len(missing), missing))
    if extra:
        die("IOC_MAP has %d labels absent from the medal table: %s" % (len(extra), extra))
    for label in labels:
        fixed = IOC_MAP[label]
        v = votes.get(label)
        if label in MANUAL_OVERRIDES:
            expect_code, expect_vote, _ = MANUAL_OVERRIDES[label]
            if fixed != expect_code:
                die("%s: fixed code %r != recorded manual pin %r" % (label, fixed, expect_code))
            top = v[0][0] if v else "<absent>"
            if top != expect_vote:
                log("[warn] %s: athletes majority vote is now %r (recorded %r), manual pin kept" % (label, top, expect_vote))
        else:
            if v is None:
                die("%s is in neither the athletes Team column nor MANUAL_OVERRIDES, cannot assign a code" % label)
            if fixed != v[0][0]:
                die("%s: fixed code %r != athletes majority vote %r" % (label, fixed, v[0][0]))
    n_manual = len(MANUAL_OVERRIDES)
    log("[ioc] %d fixed mappings (%d manual pins) cross-checked against the athletes majority vote"
        % (len(IOC_MAP), n_manual))


HOST_LABEL_OVERRIDES = {
    "United Kingdom": "Great Britain",
}


def load_hosts(labels_by_year):
    path = os.path.join(DATA_DIR, "summerOly_hosts.csv")
    h = pd.read_csv(path).dropna(subset=["Host"]).copy()
    h["Host"] = h["Host"].astype(str).str.replace(r"\(.*?\)", "", regex=True).str.strip()

    records, skipped = [], []
    for _, r in h.iterrows():
        host = r["Host"]
        if host.lower().startswith("cancelled"):
            skipped.append((int(r["Year"]), host))
            continue
        parts = [p.strip() for p in host.split(",")]
        country = parts[-1]
        city = parts[0] if len(parts) > 1 else ""
        records.append((int(r["Year"]), country, city))
    log("[hosts] %d valid host records, %d cancelled editions skipped: %s"
        % (len(records), len(skipped), [y for y, _ in skipped]))

    medal_years = set(labels_by_year)
    host_years = set(y for y, _, _ in records)
    if not medal_years.issubset(host_years):
        die("no host record in the hosts file for these editions: %s" % sorted(medal_years - host_years))
    for y in sorted(host_years - medal_years):
        log("[hosts] %d has host records but no medal rows (cancelled/future edition), ignored: %s"
            % (y, [c for yy, c, _ in records if yy == y]))

    host_flag = {}
    for y, country, _city in records:
        if y not in medal_years:
            continue
        label = HOST_LABEL_OVERRIDES.get(country, country)
        if label not in labels_by_year[y]:
            die("host %r of edition %d is not among that edition's medal-table country names" % (y, label))
        host_flag[y] = label
    return host_flag, records, skipped


def load_gdp():
    path = os.path.join(DATA_DIR, "historical_gdp_population.csv")
    g = pd.read_csv(path)
    need = {"country", "year", "population", "gdp_per_capita_2011usd", "gdp_total_2011usd_million"}
    if not need.issubset(g.columns):
        die("GDP file is missing columns: %s" % (need - set(g.columns)))
    if g.duplicated(["country", "year"]).any():
        die("GDP file has duplicate (country, year) rows")
    lut = {}
    for _, r in g.iterrows():
        lut[(r["country"], int(r["year"]))] = (
            float(r["population"]) if pd.notna(r["population"]) else np.nan,
            float(r["gdp_per_capita_2011usd"]) if pd.notna(r["gdp_per_capita_2011usd"]) else np.nan,
            float(r["gdp_total_2011usd_million"]) if pd.notna(r["gdp_total_2011usd_million"]) else np.nan,
        )
    log("[gdp] %s: %d rows / %d entities / %d-%d, gapfill values %s"
        % (os.path.basename(path), len(g), g["country"].nunique(),
           int(g["year"].min()), int(g["year"].max()), sorted(g["gapfill"].unique())))
    return lut, g


def _sum_components(vals_by_source):
    pops = np.array([v[0] for v in vals_by_source], dtype=float)
    tots = np.array([v[2] for v in vals_by_source], dtype=float)
    pop = float(pops.sum()) if np.isfinite(pops).all() else np.nan
    tot = float(tots.sum()) if np.isfinite(tots).all() else np.nan
    pc = tot * 1e6 / pop if (np.isfinite(tot) and np.isfinite(pop) and pop > 0) else np.nan
    return pop, pc, tot


def gdp_for(lut, country, year, gdp_indexed):
    if country in GDP_PROXY_RULES:
        kind, sources = GDP_PROXY_RULES[country]
        note = "%s %d: %s <- %s" % (country, year, kind, "+".join(sources))
        if kind == "proxy_sum":
            vals = [lut.get((s, year), (np.nan,) * 3) for s in sources]
            pop, pc, tot = _sum_components(vals)
        else:
            pop = pc = tot = np.nan
            for src in sources:
                p, c, t = lut.get((src, year), (np.nan,) * 3)
                if np.isfinite(t):
                    pop, pc, tot, note = p, c, t, "%s %d: %s <- %s" % (country, year, kind, src)
                    break
        if not np.isfinite(tot):
            note += " -> source lacks gdp_total, downgraded to missing"
            return pop, pc, np.nan, "missing", note
        return pop, pc, tot, kind, note

    if year >= CARRY_MIN_YEAR:
        pop, pc, tot = lut.get((country, CARRY_FROM_YEAR), (np.nan,) * 3)
        if not np.isfinite(tot):
            return pop, pc, np.nan, "missing", "%s %d: no %d source value, downgraded to missing" % (country, year, CARRY_FROM_YEAR)
        return pop, pc, tot, "carry2y", "%s %d: carry2y <- %d" % (country, year, CARRY_FROM_YEAR)

    pop, pc, tot = lut.get((country, year), (np.nan,) * 3)
    if not np.isfinite(tot):
        if (country, year) in gdp_indexed:
            note = "%s %d: macro panel has a record but gdp_total itself is NaN, missing" % (country, year)
        else:
            note = "%s %d: no series for this entity in the macro panel, missing" % (country, year)
        return pop, pc, np.nan, "missing", note
    return pop, pc, tot, "raw", ""


def old_chain_numbers():
    out = []
    for fname, cname in OLD_CHAIN_FILES:
        path = os.path.join(GEN_DIR, fname)
        if not os.path.exists(path):
            out.append((fname, None, None, None, False, 0))
            continue
        d = pd.read_csv(path)
        gbr = d[(d[cname] == "Great Britain") & (d["Year"] == 2012)]
        ire = d[(d[cname] == "Ireland") & (d["Year"] == 2012)]
        out.append((
            fname,
            int(gbr["Total"].sum()) if len(gbr) else None,
            int(ire["Host_Flag"].max()) if len(ire) and "Host_Flag" in d else None,
            int(d["Year"].nunique()),
            bool({1932, 1960} & set(d["Year"].unique())),
            int(d["Total"].sum()),
        ))
    return out


def main():
    raw = load_medals()
    labels = sorted(raw["country"].unique())
    verify_ioc_map(labels, build_athletes_vote())

    labels_by_year = {int(y): set(grp["country"]) for y, grp in raw.groupby("Year")}
    host_flag, host_records, host_skipped = load_hosts(labels_by_year)
    lut, gdp_df = load_gdp()
    gdp_indexed = set(zip(gdp_df["country"], gdp_df["year"].astype(int)))

    rows, notes, missing_detail = [], [], []
    for _, r in raw.iterrows():
        country, year = r["country"], int(r["Year"])
        pop, pc, tot, gap, note = gdp_for(lut, country, year, gdp_indexed)
        rows.append({
            "country": country,
            "ioc": IOC_MAP[country],
            "year": year,
            "gold": int(r["Gold"]),
            "silver": int(r["Silver"]),
            "bronze": int(r["Bronze"]),
            "total": int(r["Total"]),
            "host_flag": 1 if host_flag.get(year) == country else 0,
            "gdp_total_2011usd_million": tot,
            "gdp_per_capita_2011usd": pc,
            "population": pop,
            "gdp_gapfill": gap,
        })
        if note:
            notes.append(note)
        if gap == "missing":
            missing_detail.append((country, year, pop, pc, tot))

    panel = pd.DataFrame(rows, columns=PANEL_COLS)
    panel = panel.sort_values(
        ["year", "total", "gold", "silver", "bronze", "country"],
        ascending=[True, False, False, False, False, True]).reset_index(drop=True)

    if len(panel) != len(raw):
        die("assertion 1 failed: panel has %d rows != %d source rows" % (len(panel), len(raw)))
    if panel.duplicated(["country", "year"]).any():
        die("assertion 1 failed: duplicate (country, year) rows in the panel")
    if set(zip(panel["country"], panel["year"])) != set(zip(raw["country"], raw["Year"])):
        die("assertion 1 failed: (country, year) set differs from the source")
    if not panel["ioc"].apply(lambda x: x == "" or len(x) == 3).all():
        die("assertion 1 failed: ioc column has values that are neither empty nor 3-letter codes")
    if panel["ioc"].isna().any():
        die("assertion 1 failed: NaN in the ioc column (empty string expected)")
    no_ioc = sorted(set(panel.loc[panel["ioc"] == "", "country"]))
    log("[assert] 1 passed: %d rows == %d source rows, no (country, year) duplicates; %d rows without an IOC code -> %s"
        % (len(panel), len(raw), sum(1 for c in panel["ioc"] if c == ""), no_ioc))

    raw_by_year = raw.groupby("Year")["Total"].sum()
    new_by_year = panel.groupby("year")["total"].sum()
    mism = [int(y) for y in raw_by_year.index if raw_by_year[y] != new_by_year[y]]
    if mism:
        die("assertion 2 failed: totals not conserved for these editions: %s" % mism)
    if int(raw_by_year.sum()) != int(new_by_year.sum()):
        die("assertion 2 failed: global total %d != %d" % (raw_by_year.sum(), new_by_year.sum()))
    colmap = {"gold": "Gold", "silver": "Silver", "bronze": "Bronze"}
    for c, cc in colmap.items():
        if int(raw[cc].sum()) != int(panel[c].sum()):
            die("assertion 2 failed: %s not conserved" % c)
    log("[assert] 2 passed: %d/%d editions conserved per edition plus globally, sum(total)=%d "
        "(gold %d / silver %d / bronze %d)"
        % (len(raw_by_year), len(new_by_year), int(new_by_year.sum()),
           int(panel["gold"].sum()), int(panel["silver"].sum()), int(panel["bronze"].sum())))

    for y in sorted(labels_by_year):
        n_recs = sum(1 for yy, _, _ in host_records if yy == y)
        flagged = panel.loc[(panel["year"] == y) & (panel["host_flag"] == 1), "country"].tolist()
        if len(flagged) != n_recs:
            die("assertion 3 failed: edition %d has %d host_flag=1 rows vs %d records in the hosts file"
                % (y, len(flagged), n_recs))
        for c in flagged:
            if c != host_flag[y]:
                die("assertion 3 failed: edition %d flags wrong host %r, expected %r" % (y, c, host_flag[y]))
    expected = panel["year"].map(lambda y: host_flag.get(int(y), "")).values
    wrong = panel[(panel["host_flag"] == 1) & (panel["country"].values != expected)]
    if len(wrong):
        die("assertion 3 failed: rows flagged 1 that are not the host of their edition:\n%s" % wrong)
    hf_counts = {int(y): len(g) for y, g in panel[panel["host_flag"] == 1].groupby("year")}
    log("[assert] 3 passed: %d editions each with one host_flag=1 row; 2012 -> %s, 2016 -> %s, 1980 -> %s, 1972 -> %s"
        % (len(hf_counts), host_flag.get(2012), host_flag.get(2016), host_flag.get(1980), host_flag.get(1972)))

    dist = panel["gdp_gapfill"].value_counts().reindex(GAPFILL_VALUES).fillna(0).astype(int)
    unknown = set(panel["gdp_gapfill"]) - set(GAPFILL_VALUES)
    if unknown:
        die("assertion 4 failed: invalid gapfill values %s" % unknown)
    for kind in GAPFILL_PROXIES:
        sub = panel.loc[panel["gdp_gapfill"] == kind, "gdp_total_2011usd_million"]
        if sub.isna().any():
            die("assertion 4 failed: %s rows with missing gdp_total (should be downgraded to missing)" % kind)
    log("[assert] 4 passed: gdp_gapfill distribution " + ", ".join("%s=%d" % (k, v) for k, v in dist.items()))

    os.makedirs(GEN_DIR, exist_ok=True)
    panel.to_csv(PANEL_OUT, index=False)
    log("[write] %s (%d rows x %d cols)" % (PANEL_OUT, len(panel), len(panel.columns)))
    write_report(panel, raw, dist, no_ioc, host_flag, host_records, host_skipped,
                 missing_detail, new_by_year, gdp_df)
    log("[write] %s" % REPORT_OUT)

    log("\n[preview]\n" + panel.head(8).to_string())
    log("\n[preview] 2024 rows missing GDP:\n"
        + panel.loc[(panel["year"] == 2024) & (panel["gdp_gapfill"] == "missing"),
                    ["country", "ioc", "total", "gdp_gapfill"]].to_string(index=False))
    return panel


def write_report(panel, raw, dist, no_ioc, host_flag, host_records, host_skipped,
                 missing_detail, new_by_year, gdp_df):
    L = []
    A = L.append
    n_years = int(panel["year"].nunique())
    n_countries = int(panel["country"].nunique())
    total_sum = int(panel["total"].sum())

    A("# panel_clean.csv rebuild report")
    A("")
    A("- Generator: `code/build_panel.py` (any failed assertion aborts the run)")
    A("- Sources (read-only, unmodified): `data/summerOly_medal_counts.csv` (1435 rows), "
      "`summerOly_athletes.csv`, `summerOly_hosts.csv`, `historical_gdp_population.csv`")
    A("- Output: `gen/panel_clean.csv`")
    A("")

    A("## 1. Scale")
    A("")
    A("| Metric | Value |")
    A("|---|---|")
    A("| Panel rows | **%d** (strictly 1:1 with medal_counts, assertion passed) |" % len(panel))
    A("| Editions | **%d** (%d-%d, including 1932 and 1960, none missing) |"
      % (n_years, int(panel["year"].min()), int(panel["year"].max())))
    A("| Country labels | **%d** |" % n_countries)
    A("| sum(total) | **%d** (equals sum(Total) of the raw table, per-edition and global assertions passed) |" % total_sum)
    qmask = raw["NOC"].str.endswith("?")
    q_years = sorted(int(y) for y in raw.loc[qmask, "Year"].unique())
    A("| '?'-suffixed rows | %d rows / %d labels (editions %s), normalized by stripping the suffix, never summed with any other row |"
      % (int(qmask.sum()), int(raw.loc[qmask, "NOC"].nunique()),
         ", ".join(str(y) for y in q_years)))
    A("| Rows without an IOC code | **%d** rows / %d labels (%s; 1896-1904 mixed teams, not NOCs) |"
      % (int((panel["ioc"] == "").sum()), len(no_ioc), no_ioc))
    A("| Edition list | %s |" % ", ".join(str(int(y)) for y in sorted(panel["year"].unique())))
    A("")
    A("The legacy chain lost 72 rows across the 1932/1960 editions during normalization and "
      "re-summed multiple rows of the same country and year; the rebuilt panel passes rows "
      "through one-to-one with no aggregation.")
    A("")

    A("## 2. GDP coverage by edition")
    A("")
    A("`raw` = native observation in the macro panel; `proxy_*` = proxy (deu/sum/hist); "
      "`carry2y` = carried forward from 2022; `missing` = gdp_total absent for the row. "
      "Covered rows = rows with non-missing gdp_total.")
    A("")
    A("| Edition | Countries | raw | proxy_deu | proxy_sum | proxy_hist | carry2y | missing | with GDP total | Coverage |")
    A("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for y, g in panel.groupby("year"):
        c = g["gdp_gapfill"].value_counts()
        have = int(g["gdp_total_2011usd_million"].notna().sum())
        A("| %d | %d | %d | %d | %d | %d | %d | %d | %d | %.1f%% |"
          % (int(y), len(g), int(c.get("raw", 0)), int(c.get("proxy_deu", 0)),
             int(c.get("proxy_sum", 0)), int(c.get("proxy_hist", 0)),
             int(c.get("carry2y", 0)), int(c.get("missing", 0)), have, 100.0 * have / len(g)))
    have = int(panel["gdp_total_2011usd_million"].notna().sum())
    A("| **Total** | **%d** | **%d** | **%d** | **%d** | **%d** | **%d** | **%d** | **%d** | **%.1f%%** |"
      % (len(panel), int(dist["raw"]), int(dist["proxy_deu"]), int(dist["proxy_sum"]),
         int(dist["proxy_hist"]), int(dist["carry2y"]), int(dist["missing"]), have,
         100.0 * have / len(panel)))
    A("")
    A("### gdp_gapfill distribution over the whole panel (assertion 4)")
    A("")
    A("| gapfill | rows | share |")
    A("|---|---:|---:|")
    for k in GAPFILL_VALUES:
        A("| %s | %d | %.1f%% |" % (k, int(dist[k]), 100.0 * dist[k] / len(panel)))
    A("")

    A("## 3. Legacy chain vs rebuilt panel")
    A("")
    old = old_chain_numbers()
    A("| Chain file | GBR 2012 total | Ireland 2012 host_flag | Editions | Has 1932/1960 | sum(Total) |")
    A("|---|---:|---:|---:|---|---:|")
    for fname, gbr, ire, nyrs, has_both, osum in old:
        if gbr is None:
            A("| legacy `%s` | (file missing) | - | - | - | - |" % fname)
            continue
        A("| legacy `%s` | %s | %s | %s | %s | %s |"
          % (fname, gbr, ire, nyrs, "yes" if has_both else "**no (editions lost)**", osum))
    gbr_new = int(panel.loc[(panel["country"] == "Great Britain") & (panel["year"] == 2012), "total"].iloc[0])
    ire_new = int(panel.loc[(panel["country"] == "Ireland") & (panel["year"] == 2012), "host_flag"].iloc[0])
    A("| **rebuilt `panel_clean.csv`** | **%d** | **%d** | **%d** | **yes** | **%d** |"
      % (gbr_new, ire_new, n_years, total_sum))
    A("")
    ref = old[1] if len(old) > 1 and old[1][1] is not None else old[0]
    ire_old = ref[2] if ref[2] is not None else "n/a"
    A("- GBR 2012 total: legacy **%s** -> rebuilt **%d** (the legacy chain summed 5 duplicate rows: 65 x 5)"
      % (ref[1], gbr_new))
    A("- Ireland 2012 host_flag: legacy **%s** -> rebuilt **%d** (the 2012 host was Great Britain; "
      "Ireland has host_flag 0 in every edition)" % (ire_old, ire_new))
    A("- Editions 1932/1960: legacy 0 rows -> rebuilt %d rows (%d + %d), both editions restored"
      % (int((panel["year"] == 1932).sum() + (panel["year"] == 1960).sum()),
         int((panel["year"] == 1932).sum()), int((panel["year"] == 1960).sum())))
    A("- sum(Total): legacy **%d** -> rebuilt **%d** (= row-by-row total of the raw table; the legacy chain inflated it by %.1f%%)"
      % (ref[5], total_sum, 100.0 * (ref[5] / total_sum - 1)))
    A("- Editions: legacy **%s** -> rebuilt **%d**" % (ref[3], n_years))
    A("")

    A("## 4. Host flagging")
    A("")
    A("`summerOly_hosts.csv` has %d records, of which %d cancelled editions (%s) generate no row; "
      "2028/2032 have no medal rows and are not in the panel. Exactly one row per edition, its "
      "host, gets `host_flag=1`; assertion: host_flag=1 row count == host records for that "
      "edition (passed for all %d editions)."
      % (len(host_records) + len(host_skipped), len(host_skipped),
         ", ".join(str(y) for y, _ in host_skipped), n_years))
    A("")
    A("| Edition | Host label | host_flag=1 rows |")
    A("|---:|---|---:|")
    for y in sorted(host_flag):
        A("| %d | %s | %d |" % (y, host_flag[y], hf_count(panel, y)))
    A("")

    A("## 5. GDP proxies and missing values")
    A("")
    A("### 5.1 Proxy rules in force")
    A("")
    rules_text = {
        "proxy_deu": "East/West Germany and the joint German team -> the unified Germany (DEU) series: Maddison provides a single unified-Germany series, the only available and consistent source",
        "proxy_sum": "Joint delegations -> sum of member economies; the Unified Team = 12 of the 15 ex-Soviet republics (Baltic states competed separately); Serbia and Montenegro and FR Yugoslavia are both Serbia+Montenegro; Australasia = Australia+New Zealand",
        "proxy_hist": "Historical predecessor entities -> successor series: Bohemia->CZE (falling back to Czechoslovakia for 1900/1908, since the CZE panel starts in 1950), Russian Empire->RUS",
        "carry2y": "2024 edition -> carry the 2022 value: the macro panel ends in 2022 and Paris 2024 has no contemporaneous observation; Tokyo 2020 keeps its raw value",
    }
    for k in ("proxy_deu", "proxy_sum", "proxy_hist", "carry2y"):
        sub = panel[panel["gdp_gapfill"] == k]
        if not len(sub):
            continue
        A("- **%s** (%d rows, labels: %s): %s"
          % (k, len(sub), ", ".join(sorted(sub["country"].unique())), rules_text[k]))
    A("")
    A("`missing` covers two cases: (a) no series at all for the entity in the macro panel; "
      "(b) an explicit downgrade because a proxy/carry source lacks gdp_total for the required "
      "year. Every row is listed in 5.2.")
    A("")
    A("### 5.2 List of missing rows (nothing silent)")
    A("")
    A("%d rows / %d labels in total." % (len(missing_detail), len(set(c for c, *_ in missing_detail))))
    A("")
    A("| country | year | population | gdp_per_capita | gdp_total | Reason missing |")
    A("|---|---:|---:|---:|---:|---|")
    gdp_pairs = set(zip(gdp_df["country"], gdp_df["year"].astype(int)))
    gdp_countries = set(gdp_df["country"])
    for c, y, p, pc, t in sorted(missing_detail):
        if c in GDP_PROXY_RULES:
            kind, srcs = GDP_PROXY_RULES[c]
            why = "proxy source (%s) %s lacks gdp_total in %d, downgraded" % (
                kind, "+".join(srcs) if kind == "proxy_sum" else "/".join(srcs), y)
        elif y >= CARRY_MIN_YEAR:
            why = "no %d value in the macro panel, downgraded" % CARRY_FROM_YEAR
        elif (c, y) in gdp_pairs:
            why = "macro panel has a record for this (entity, year) but gdp_total itself is NaN (not covered by Maddison)"
        elif c in gdp_countries:
            s = gdp_df.loc[gdp_df["country"] == c, "year"]
            why = "entity has a series in the macro panel but no %d observation (panel starts %d or has gaps)" % (y, int(s.min()))
        else:
            why = "no series for this entity in the macro panel (not covered by Maddison)"
        fmt = lambda v: ("NaN" if (v is None or not np.isfinite(v)) else "%.0f" % v)
        A("| %s | %d | %s | %s | %s | %s |" % (c, y, fmt(p), fmt(pc), fmt(t), why))
    A("")
    A("Note: some missing rows still carry single-column observations (population or gdp_per_capita) "
      "in the macro panel, kept as-is in their columns; since gdp_total is missing, `gdp_gapfill` "
      "is uniformly `missing` -- no substituting neighbouring years, no interpolation.")
    A("")

    A("## 6. Processing rules")
    A("")
    A("1. All %d '?'-suffixed rows are normalized by stripping the suffix, never summed with any "
      "other row; an assertion checks that no (country, year) duplicates remain." % int(qmask.sum()))
    A("2. `country_mapping.csv` (a 1060-entry dump of raw names, including horse and ship names, "
      "many-to-one) is **not used**.")
    A("3. IOC codes are built from the Team->NOC majority vote in `summerOly_athletes.csv` with "
      "%d manual pins; they are fixed in `build_panel.py` as `IOC_MAP` (%d entries) and "
      "`MANUAL_OVERRIDES` (%d entries); every run recomputes the vote and cross-checks it "
      "against the fixed table (any mismatch aborts)."
      % (len(MANUAL_OVERRIDES), len(IOC_MAP), len(MANUAL_OVERRIDES)))
    A("4. GDP is matched exactly on (country, year), with no fuzzy matching and no interpolation; "
      "the only time extrapolation is the explicit `carry2y` of 2022 into 2023/2024 (Tokyo 2020 "
      "stays raw).")
    A("5. `host_flag` comes only from the edition's host in `summerOly_hosts.csv`; the cancelled "
      "1916/1940/1944 editions and the future 2028/2032 editions are not in the panel.")
    A("6. No file under `data/` is modified; the script writes only `gen/panel_clean.csv` and this report.")
    A("")

    votes = build_athletes_vote()
    A("## Appendix A: full country name -> IOC code mapping")
    A("")
    A("Source: majority NOC code by athlete-row count for `Team == country name` in "
      "`summerOly_athletes.csv`; `†` = manual pin (reason in parentheses). %d entries in total."
      % len(IOC_MAP))
    A("")
    A("| country | ioc |")
    A("|---|---|")
    for label in sorted(IOC_MAP):
        code = IOC_MAP[label]
        if label in MANUAL_OVERRIDES:
            why = MANUAL_OVERRIDES[label][2]
            A("| %s | %s † (%s) |" % (label, code if code else "(empty, not an NOC)", why))
        else:
            A("| %s | %s |" % (label, code))
    A("")
    A("### All entries where the majority vote differs from the fixed table (all manual pins)")
    A("")
    A("| country | fixed code | athletes majority vote (rows) | Reason |")
    A("|---|---|---|---|")
    for label in sorted(MANUAL_OVERRIDES):
        v = votes.get(label) or []
        sec = "; ".join("%s(%d)" % (c, n) for c, n in v) if v else "(Team name not in the athletes table)"
        A("| %s | %s | %s | %s |"
          % (label, IOC_MAP[label] if IOC_MAP[label] else "(empty)", sec, MANUAL_OVERRIDES[label][2]))
    A("")

    with open(REPORT_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


def hf_count(panel, year):
    return int(((panel["year"] == year) & (panel["host_flag"] == 1)).sum())


if __name__ == "__main__":
    main()
