# 실행: python code/ch03_prompt.py
import sys, pathlib, json
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from common import get_genai_client, GEMINI_MODEL, DATA
from google.genai import types
import pandas as pd

client = get_genai_client()
# cs_inquiries.csv = 고객 문의 60건. category_hint 컬럼이 정답 라벨(정확도 측정용).
df = pd.read_csv(DATA / "cs_inquiries.csv")
print("문의 건수:", len(df))
print(df[["content", "category_hint"]].head(3).to_string(index=False))


ROLE = (
    "너는 승승장구몰의 친절한 CS 상담원이다. "
    "고객 문의에 존댓말로 공감하며 간결하게 답하라. "
    "확실하지 않은 정보는 '확인 후 안내드리겠습니다'라고 답하라."
)

def reply(content: str) -> str:
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"고객 문의: {content}",
        config=types.GenerateContentConfig(system_instruction=ROLE, temperature=0.3),
    )
    return resp.text

sample = df.iloc[0]["content"]
print("문의:", sample)
print("답변:", reply(sample))


CATEGORIES = ["배송", "환불", "교환", "결제", "상품문의", "칭찬", "불만"]

FEWSHOT = """다음 고객 문의를 아래 7개 중 정확히 하나로 분류하라.
카테고리: 배송 / 환불 / 교환 / 결제 / 상품문의 / 칭찬 / 불만
카테고리 이름 한 단어만 출력하라(다른 말 금지).

[예시]
문의: 반품하면 배송비는 누가 부담하나요?           → 환불
문의: 색상이 사진과 달라요. 다른 색으로 바꿔주세요.  → 교환
문의: 상담원분이 정말 친절하셨어요. 감사합니다.      → 칭찬
문의: 카드가 두 번 청구됐어요.                      → 결제
"""

def classify(content: str) -> str:
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=f"{FEWSHOT}\n[분류할 문의]\n문의: {content} →",
        config=types.GenerateContentConfig(temperature=0),
    )
    out = resp.text.strip()
    for c in CATEGORIES:
        if c in out:
            return c
    return "기타"

# 60건 전부 분류 → 정답 라벨(category_hint)과 비교
df["pred"] = df["content"].head(5).apply(classify)
correct = (df["pred"] == df["category_hint"]).sum()
print(f"분류 정확도: {correct/len(df):.1%}  ({correct}/{len(df)})")

# 틀린 사례 몇 개 출력
wrong = df[df["pred"] != df["category_hint"]]
if len(wrong):
    print("틀린 사례(일부):")
    print(wrong[["content", "category_hint", "pred"]].head().to_string(index=False))


def triage(content: str) -> dict:
    resp = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=(
            "다음 고객 문의를 분석해 JSON으로만 답하라.\n"
            "키: category(배송/환불/교환/결제/상품문의/칭찬/불만 중 하나), "
            "urgent(true/false), summary(20자 이내 한국어)\n"
            f"문의: {content}"
        ),
        config=types.GenerateContentConfig(
            temperature=0,
            response_mime_type="application/json",
        ),
    )
    return json.loads(resp.text)

r = triage("어제 받은 제품이 박살나서 왔어요. 당장 환불해주세요!")
print(r)
print("긴급?", r["urgent"], "/ 분류:", r["category"])
