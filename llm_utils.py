
"""
AI Copilot layer — Gemini API, RAG and Agentic AI.

Components:
1. Situation Report generation
2. TF-IDF based disaster-guidance RAG
3. Agentic response agent with tool calling
4. Public safety advisory generation
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import extras_utils as ex
import report_utils as rpt

genai = None

DEFAULT_MODEL = "gemini-3.5-flash-lite"


def _ensure_genai():
    global genai
    if genai is None:
        import google.generativeai as _genai
        genai = _genai
    return genai


def get_client(api_key):
    g = _ensure_genai()
    g.configure(api_key=api_key)
    return api_key


# =========================================================
# 1. DISASTER KNOWLEDGE BASE
# =========================================================

KNOWLEDGE_BASE = [
    {
        "title": "Flood — Overview",
        "hazard": "Flood",
        "text": (
            "A flood occurs when water overflows onto land that is normally dry. "
            "Common causes include heavy rainfall, overflowing rivers, storm surges "
            "and inadequate drainage. Floods can damage buildings, roads, crops and "
            "electrical infrastructure. Flood water may contain sewage, chemicals "
            "and other contaminants. Avoid walking or driving through flood water "
            "and follow official evacuation instructions."
        )
    },
    {
        "title": "Flood — Before",
        "hazard": "Flood",
        "text": (
            "Households in flood-prone areas should identify the nearest higher "
            "ground and evacuation route in advance. Keep important documents in "
            "a waterproof pouch, store drinking water for at least three days, "
            "and elevate electrical appliances where possible. Community early-warning "
            "systems, including sirens and SMS alerts, should be tested regularly."
        )
    },
    {
        "title": "Flood — During",
        "hazard": "Flood",
        "text": (
            "Avoid walking or driving through moving flood water. As little as "
            "15 cm of fast-moving water can knock a person off their feet, and "
            "around 30 cm can float many vehicles. Move to higher ground when "
            "instructed. Disconnect electrical systems only if it is safe to do so. "
            "Follow official evacuation instructions."
        )
    },
    {
        "title": "Flood — After",
        "hazard": "Flood",
        "text": (
            "Do not re-enter a flooded structure until authorities confirm it is "
            "safe. Boil or treat drinking water according to local public-health "
            "guidance until the supply is declared safe. Discard food contaminated "
            "by flood water. Photograph property damage before cleanup when safe."
        )
    },
    {
        "title": "Wildfire — Overview",
        "hazard": "Wildfire",
        "text": (
            "A wildfire is an uncontrolled fire that spreads through vegetation "
            "such as forests, grasslands or shrublands. It can spread rapidly in "
            "hot, dry and windy conditions. Wildfires threaten people, homes, "
            "wildlife and air quality. Follow evacuation orders and avoid smoke "
            "and active fire areas."
        )
    },
    {
        "title": "Wildfire — Before",
        "hazard": "Wildfire",
        "text": (
            "Clear dry vegetation and flammable material near structures. Prepare "
            "an emergency bag containing medication, documents and water. Identify "
            "two evacuation routes in case one is blocked. Sign up for local "
            "fire-department alerts where available."
        )
    },
    {
        "title": "Wildfire — During",
        "hazard": "Wildfire",
        "text": (
            "If ordered to evacuate, leave immediately and follow official routes. "
            "Do not wait to see whether the fire changes direction. Avoid smoke "
            "and active fire areas. If trapped, call emergency services and follow "
            "their instructions."
        )
    },
    {
        "title": "Wildfire — After",
        "hazard": "Wildfire",
        "text": (
            "Wait for official clearance before returning home. Watch for hot spots, "
            "smouldering embers, damaged structures and downed power lines. Avoid "
            "areas with heavy smoke and follow local air-quality advice."
        )
    },
    {
        "title": "Cyclone — Overview",
        "hazard": "Cyclone",
        "text": (
            "A cyclone is a large rotating storm system that develops around a "
            "low-pressure centre. Tropical cyclones can bring destructive winds, "
            "heavy rainfall, coastal storm surges and flooding. Their impacts "
            "include damage to buildings, power networks and transport. Follow "
            "official weather warnings and evacuation instructions."
        )
    },
    {
        "title": "Cyclone — Before",
        "hazard": "Cyclone",
        "text": (
            "Secure or bring indoors loose outdoor objects. Protect windows and "
            "doors, and identify a safe interior room away from windows. Stock "
            "at least three days of food, water and essential medication. Monitor "
            "official weather forecasts and evacuation instructions."
        )
    },
    {
        "title": "Cyclone — During",
        "hazard": "Cyclone",
        "text": (
            "Stay indoors in a safe room throughout the storm. Do not go outside "
            "during a temporary lull, because it may be the eye of the cyclone. "
            "Keep mobile devices charged and follow official emergency updates."
        )
    },
    {
        "title": "Cyclone — After",
        "hazard": "Cyclone",
        "text": (
            "Treat downed power lines as live. Avoid coastal areas, flooded roads "
            "and riverbanks because storm surges and flash floods may continue "
            "after winds subside. Report structural damage and follow official "
            "instructions before returning to affected areas."
        )
    },
    {
        "title": "Earthquake — Overview",
        "hazard": "Earthquake",
        "text": (
            "An earthquake is the shaking of the Earth's surface caused by the "
            "sudden release of energy within the Earth's crust. It can damage "
            "buildings, roads and utilities, and may trigger landslides or "
            "tsunamis in some circumstances. The safest response depends on "
            "whether a person is indoors, outdoors or in a vehicle."
        )
    },
    {
        "title": "Earthquake — Before",
        "hazard": "Earthquake",
        "text": (
            "Secure heavy furniture and shelving to walls. Identify sturdy "
            "furniture that can provide cover during shaking. Store emergency "
            "supplies, including water, a torch and first aid items, in an "
            "accessible place. Practise Drop, Cover and Hold On."
        )
    },
    {
        "title": "Earthquake — During",
        "hazard": "Earthquake",
        "text": (
            "Drop to the ground, take cover under sturdy furniture if available, "
            "and hold on until shaking stops. If outdoors, move to open ground "
            "away from buildings, trees and power lines. Do not run outside "
            "while shaking is occurring."
        )
    },
    {
        "title": "Earthquake — After",
        "hazard": "Earthquake",
        "text": (
            "Check for injuries and hazards. If you smell gas or hear hissing, "
            "leave the area and contact emergency services. Expect aftershocks. "
            "Avoid damaged buildings and elevators, and follow official safety "
            "instructions before re-entering structures."
        )
    },
    {
        "title": "General — Vulnerable groups",
        "hazard": "General",
        "text": (
            "Children, older adults and people with mobility limitations may need "
            "additional assistance during disasters. Identify helpers in advance, "
            "keep required medical equipment charged and stocked, and include "
            "individual needs in the household emergency plan."
        )
    },
    {
        "title": "General — Communication",
        "hazard": "General",
        "text": (
            "Agree on an out-of-area contact person and a meeting point in case "
            "family members become separated. Keep a battery-powered or hand-crank "
            "radio available because mobile networks may become overloaded during "
            "a major disaster."
        )
    },
]


# =========================================================
# 2. TF-IDF RETRIEVAL
# =========================================================

_vectorizer = None
_doc_vectors = None


def _detect_hazard(query):
    """Detect a hazard explicitly mentioned in the question."""
    q = query.lower()

    keywords = {
        "Flood": (
            "flood", "floods", "flooding", "floodwater"
        ),
        "Wildfire": (
            "wildfire", "wildfires", "forest fire", "bushfire"
        ),
        "Cyclone": (
            "cyclone", "cyclones", "hurricane", "typhoon"
        ),
        "Earthquake": (
            "earthquake", "earthquakes", "seismic"
        ),
    }

    for hazard, terms in keywords.items():
        if any(term in q for term in terms):
            return hazard

    return None


def _normalize_hazard(hazard):
    """Normalize the hazard name used by the application."""
    if not hazard:
        return None

    value = str(hazard).strip().lower()

    aliases = {
        "flood": "Flood",
        "floods": "Flood",
        "flooding": "Flood",
        "wildfire": "Wildfire",
        "wildfires": "Wildfire",
        "forest fire": "Wildfire",
        "cyclone": "Cyclone",
        "hurricane": "Cyclone",
        "typhoon": "Cyclone",
        "earthquake": "Earthquake",
        "seismic": "Earthquake",
    }

    return aliases.get(value, str(hazard).strip().title())


def _build_index():
    global _vectorizer, _doc_vectors

    # Index title, hazard label and passage together.
    texts = [
        f"{d['title']} {d['hazard']} {d['text']}"
        for d in KNOWLEDGE_BASE
    ]

    _vectorizer = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
    )

    _doc_vectors = _vectorizer.fit_transform(texts)


def retrieve(query, hazard_filter=None, top_k=3):
    """Retrieve relevant passages with hazard-aware filtering."""
    global _vectorizer, _doc_vectors

    if _vectorizer is None or _doc_vectors is None:
        _build_index()

    # Explicit hazard in the question takes priority over
    # the currently predicted hazard shown by the application.
    query_hazard = _detect_hazard(query)
    active_hazard = query_hazard or _normalize_hazard(hazard_filter)

    q_vec = _vectorizer.transform([query])
    similarities = cosine_similarity(q_vec, _doc_vectors)[0]

    candidates = []

    for i, score in enumerate(similarities):
        doc = KNOWLEDGE_BASE[i]

        if active_hazard:
            if doc["hazard"] not in (active_hazard, "General"):
                continue

        # Never return unrelated zero-score passages.
        if score > 0:
            candidates.append((i, float(score)))

    candidates.sort(key=lambda item: item[1], reverse=True)

    return [
        (KNOWLEDGE_BASE[i], score)
        for i, score in candidates[:top_k]
    ]


# =========================================================
# 3. GEMINI TEXT GENERATION
# =========================================================

def _generate_text(
    model_name,
    prompt,
    max_tokens=1024,
    temperature=0.2,
):
    """Generate text with Gemini."""
    g = _ensure_genai()
    model = g.GenerativeModel(model_name)

    response = model.generate_content(
        prompt,
        generation_config=g.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature,
        ),
    )

    return (response.text or "").strip()


# =========================================================
# 4. SITUATION REPORT
# =========================================================

def generate_situation_report(
    client,
    model,
    pred_class,
    confidence,
    location,
    feature_values,
    shap_top_features,
    lang="en",
):
    lang_instruction = (
        "Respond entirely in Hindi."
        if lang == "hi"
        else "Respond entirely in English."
    )

    loc_str = (
        f"Lat {location[0]:.4f}, Lon {location[1]:.4f}"
        if location
        else "Not provided"
    )

    prompt = f"""
You are a disaster-response analyst.

Predicted hazard: {pred_class}
Model confidence: {confidence:.1f}%
Location: {loc_str}
Environmental readings: {feature_values}
Top contributing factors (SHAP): {shap_top_features}

Write a concise professional situation report containing:
1. Situation summary.
2. Key contributing factors in plain language.
3. One recommended next step.

Write 2-3 short paragraphs, 100-140 words in total.
Do not use markdown headings or bullet symbols.
{lang_instruction}
"""

    return _generate_text(model, prompt, max_tokens=400)


# =========================================================
# 5. RAG — GROUNDED QUESTION AND ANSWER
# =========================================================

def rag_answer(client, model, query, hazard_class, lang="en"):
    results = retrieve(
        query,
        hazard_filter=hazard_class,
        top_k=3,
    )

    if not results:
        message = (
            "Diye gaye documents mein is question ka relevant "
            "answer nahi mila."
            if lang == "hi"
            else "I could not find relevant information in the "
                 "provided documents."
        )
        return message, []

    context = "\n\n".join(
        f"[{doc['title']}]\n{doc['text']}"
        for doc, score in results
    )

    lang_instruction = (
        "Answer entirely in Hindi."
        if lang == "hi"
        else "Answer entirely in English."
    )

    prompt = f"""
You are a precise disaster-safety assistant.

Answer the user's question directly using the context below.

Rules:
- Start directly with the answer.
- Do not begin with "Based on the provided context".
- Do not begin with "According to the provided context".
- Do not add an introduction about the context or documents.
- Use only relevant facts from the supplied context.
- If the context does not contain the requested information,
  clearly say that it is not available in the documents.
- Do not invent facts or claim that a passage says something
  it does not say.
- Give practical, concise information.
- Do not add unrelated disaster guidance.

{lang_instruction}

Context:
{context}

Question: {query}

Answer:
"""

    answer = _generate_text(
        model,
        prompt,
        max_tokens=350,
        temperature=0.2,
    )

    sources = [doc["title"] for doc, score in results]

    return answer, sources


# =========================================================
# 6. AGENTIC AI — TOOL DECLARATIONS
# =========================================================

def _tool_declarations():
    g = _ensure_genai()

    return [
        g.protos.FunctionDeclaration(
            name="search_knowledge_base",
            description=(
                "Search disaster guidance for a query and hazard type."
            ),
            parameters=g.protos.Schema(
                type=g.protos.Type.OBJECT,
                properties={
                    "query": g.protos.Schema(
                        type=g.protos.Type.STRING,
                        description="What to search for",
                    ),
                    "hazard": g.protos.Schema(
                        type=g.protos.Type.STRING,
                        description=(
                            "One of Flood, Wildfire, Cyclone, Earthquake"
                        ),
                    ),
                },
                required=["query", "hazard"],
            ),
        ),
        g.protos.FunctionDeclaration(
            name="get_live_weather",
            description=(
                "Fetch live weather data for a latitude and longitude."
            ),
            parameters=g.protos.Schema(
                type=g.protos.Type.OBJECT,
                properties={
                    "lat": g.protos.Schema(
                        type=g.protos.Type.NUMBER
                    ),
                    "lon": g.protos.Schema(
                        type=g.protos.Type.NUMBER
                    ),
                },
                required=["lat", "lon"],
            ),
        ),
        g.protos.FunctionDeclaration(
            name="get_recommended_actions",
            description=(
                "Get recommended emergency actions for a hazard."
            ),
            parameters=g.protos.Schema(
                type=g.protos.Type.OBJECT,
                properties={
                    "hazard": g.protos.Schema(
                        type=g.protos.Type.STRING
                    ),
                },
                required=["hazard"],
            ),
        ),
    ]


def _execute_tool(name, tool_input):
    if name == "search_knowledge_base":
        results = retrieve(
            tool_input["query"],
            hazard_filter=tool_input.get("hazard"),
            top_k=2,
        )

        if not results:
            return "No relevant passages found."

        return "\n\n".join(
            f"{doc['title']}: {doc['text']}"
            for doc, score in results
        )

    elif name == "get_live_weather":
        weather = ex.fetch_live_weather(
            tool_input["lat"],
            tool_input["lon"],
        )

        return (
            str(weather)
            if weather
            else "Weather data unavailable."
        )

    elif name == "get_recommended_actions":
        hazard = _normalize_hazard(tool_input.get("hazard"))
        info = rpt.HAZARD_ACTIONS.get(hazard, {})

        return (
            f"Severity: {info.get('severity', 'N/A')}. "
            f"Actions: {info.get('actions', [])}"
        )

    return "Unknown tool."


# =========================================================
# 7. AGENTIC RESPONSE AGENT
# =========================================================

def run_response_agent(
    client,
    model,
    pred_class,
    confidence,
    location,
    lang="en",
    max_steps=5,
):
    lang_instruction = (
        "Write the final alert entirely in Hindi."
        if lang == "hi"
        else "Write the final alert entirely in English."
    )

    loc_str = (
        f"latitude {location[0]}, longitude {location[1]}"
        if location
        else "an unspecified location"
    )

    system = (
        "You are a disaster-response assistant. "
        f"A model detected a potential {pred_class} hazard "
        f"with {confidence:.1f}% confidence at {loc_str}. "
        "Use available tools when appropriate to gather relevant "
        "guidance, weather information and recommended actions. "
        "Do not claim that a tool was used unless it was actually called. "
        "Write a short, actionable alert of 45-60 words. "
        f"{lang_instruction} "
        "Respond with only the final alert text, without a preamble."
    )

    g = _ensure_genai()

    gemini_model = g.GenerativeModel(
        model,
        system_instruction=system,
        tools=[
            g.protos.Tool(
                function_declarations=_tool_declarations()
            )
        ],
        generation_config=g.types.GenerationConfig(
            max_output_tokens=600,
            temperature=0.2,
        ),
    )

    chat = gemini_model.start_chat()
    trace = []
    message = "Investigate this hazard and draft an alert."

    for _ in range(max_steps):
        response = chat.send_message(message)
        candidate = response.candidates[0]

        function_calls = [
            part.function_call
            for part in candidate.content.parts
            if (
                getattr(part, "function_call", None)
                and part.function_call.name
            )
        ]

        if not function_calls:
            final_text = "".join(
                part.text
                for part in candidate.content.parts
                if getattr(part, "text", None)
            )

            return trace, final_text.strip()

        response_parts = []

        for function_call in function_calls:
            tool_input = dict(function_call.args)

            output = _execute_tool(
                function_call.name,
                tool_input,
            )

            trace.append({
                "tool": function_call.name,
                "input": tool_input,
                "output": output,
            })

            response_parts.append(
                g.protos.Part(
                    function_response=g.protos.FunctionResponse(
                        name=function_call.name,
                        response={"result": output},
                    )
                )
            )

        message = g.protos.Content(parts=response_parts)

    return trace, (
        "The agent could not complete the response within "
        "the configured step limit."
    )


# =========================================================
# 8. GENERATIVE PUBLIC SAFETY ADVISORY
# =========================================================

def generate_public_advisory(
    client,
    model,
    pred_class,
    confidence,
    location,
    lang="en",
):
    lang_instruction = (
        "Write entirely in Hindi."
        if lang == "hi"
        else "Write entirely in English."
    )

    loc_part = (
        " near the specified location"
        if location
        else ""
    )

    prompt = f"""
Write a public safety advisory for a detected {pred_class}
risk with model confidence {confidence:.1f}%{loc_part}.

Requirements:
- Write 40-55 words.
- State the detected hazard.
- Give two concrete safety actions.
- Use exactly one relevant emoji.
- Do not use hashtags or markdown.
- Do not invent live conditions or claim an official warning
  exists unless that information is supplied.
{lang_instruction}
"""

    return _generate_text(
        model,
        prompt,
        max_tokens=250,
        temperature=0.2,
    )
