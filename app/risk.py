import re
PATTERNS=[r"chest pain.*(shortness of breath|faint|sweat)",r"face droop|one.side weakness|slurred speech",r"cannot breathe|severe difficulty breathing",r"suicid(al|e)|kill myself|self.harm",r"heavy bleeding|uncontrolled bleeding",r"overdose|poison(ed|ing)",r"anaphylaxis|throat.*closing"]
def assess_risk(case):
    text=case["conversation_text"].lower(); match=next((p for p in PATTERNS if re.search(p,text,re.S)),None)
    return {"possible_emergency":bool(match),"matched_pattern":match or "","instruction":"Urgent action must appear in the first two sentences; do not delay for questions." if match else "Use calibrated urgency."}

