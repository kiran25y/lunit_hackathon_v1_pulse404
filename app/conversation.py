import re

def build_case_state(messages):
    user = [m.get("content","") for m in messages if m.get("role") == "user"]
    all_text = "\n".join(m.get("content","") for m in messages)
    last = user[-1] if user else ""
    return {"current_request":last,"conversation_text":all_text,"turns":len(messages),"language_hint":"ko" if re.search(r"[가-힣]",last) else "en","professional_hint":bool(re.search(r"\b(patient|assessment|differential|ICD|KCD|dose|mg|lab|clinical)\b",last,re.I)),"reference_risk":bool(re.search(r"\b(it|that|this|its|they|them)\b|그것|이것|그 약",last,re.I))}

def self_contained_query(query, case):
    return f"Current request: {query}\nRelevant conversation:\n{case['conversation_text'][-6000:]}"

