"""Read the ground-truth Excel logs (data/videos/<rec>/<rec>_ground_truth_ai_app.xlsx) without extra packages.

An .xlsx is a zip of XML files; formula cells (Pr_Tare_KG) keep their last calculated value, which we read.
A repeated column name gets a suffix like pandas does (14_43: Time, Time.1). Excel times become seconds since
midnight.

    rows = read_log("data/videos/14_43/14_43_ground_truth_ai_app.xlsx")
    bottles = [r for r in rows if r["Primagaz_status"] is not None]   # rows with only a weight are no bottle
"""
import re
import zipfile
import xml.etree.ElementTree as ET

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"


def read_log(path):
    """List of dicts, one per sheet row after the header, with 'xlsx_row' = the row number in Excel."""
    z = zipfile.ZipFile(path)
    strings = ["".join(t.text or "" for t in si.iter(NS + "t"))
               for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall(NS + "si")]
    cells = {}
    for row in ET.fromstring(z.read("xl/worksheets/sheet1.xml")).iter(NS + "row"):
        values = {}
        for c in row.findall(NS + "c"):
            v = c.find(NS + "v")
            if v is not None:
                values[re.match(r"[A-Z]+", c.get("r")).group()] = \
                    strings[int(v.text)] if c.get("t") == "s" else float(v.text)
        cells[int(row.get("r"))] = values

    header = cells.pop(1)
    cols = sorted(header, key=lambda c: (len(c), c))
    names, seen = {}, {}
    for c in cols:
        n = header[c]
        names[c] = n if n not in seen else f"{n}.{seen[n]}"
        seen[n] = seen.get(n, 0) + 1

    rows = []
    for r in range(2, max(cells) + 1):
        row = {"xlsx_row": r}
        for c in cols:
            v = cells.get(r, {}).get(c)
            if names[c].startswith("Time") and isinstance(v, float):
                v = round((v % 1) * 86400, 1)  # Excel date -> seconds since midnight
            row[names[c]] = v
        rows.append(row)
    return rows
