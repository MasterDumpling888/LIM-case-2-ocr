from langchain_core.prompts import ChatPromptTemplate

RECTIFICATION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert in document processing and OCR correction.
Your task is to rectify malformed text extracted from receipts.
- Correct spelling errors (e.g., "sollor" -> "seller", "walmrt" -> "walmart").
- Complete compound entities with missing components (e.g., "McDon" -> "McDonald's").
- Normalize text while preserving original meaning.
Return ONLY the corrected text."""),
    ("human", "{ocr_text}")
])

EXTRACTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert in financial document extraction.
Extract the following entities from the provided receipt text into a JSON format:
- company: The name of the vendor/company.
- date: The date of the transaction (normalize to DD/MM/YYYY if possible).
- address: The full address of the vendor.
- subtotal: The subtotal amount before tax.
- tax: The tax amount.
- total: The total amount paid.

RULES:
1. Extract values EXACTLY as they appear in the text. 
2. DO NOT calculate or invent numbers to make the math work. If a value is missing or unclear, use an empty string.
3. If multiple numbers look like a "Total", pick the one explicitly labeled "TOTAL" or "AMOUNT DUE".
4. Return ONLY the JSON object."""),
    ("human", "{text}")
])

VISION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert in visual document analysis.
Analyze the provided receipt image and extract the following entities into a JSON format:
- company: The name of the vendor/company.
- date: The date of the transaction (normalize to DD/MM/YYYY if possible).
- address: The full address of the vendor.
- subtotal: The subtotal amount before tax.
- tax: The tax amount.
- total: The total amount paid.

RULES:
1. Extract values EXACTLY as they appear on the receipt.
2. DO NOT calculate or invent numbers to make the math work. If a value is not visible, use an empty string.
3. Return ONLY the JSON object."""),
    ("human", [
        {"type": "text", "text": "Extract entities from this receipt image."},
        {"type": "image_url", "image_url": "{image_url}"}
    ])
])

QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a helpful assistant that answers questions about a receipt based on its extracted content.
Extracted Content:
{extracted_content}

Answer the user's question concisely based ONLY on the provided content."""),
    ("human", "{question}")
])
