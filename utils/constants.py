GPT_35_TURBO_MODEL = "gpt-3.5-turbo-0125"
GPT_4O_MODEL = "gpt-4o"
O1_MINI_MODEL = "o1-mini"
GPT_4O_MINI = "gpt-4o-mini"
MIXTRAL_MODEL = "mixtral-8x7b-32768"
LLAMA_8_MODEL = "llama3-8b-8192"
LLAMA_8_TOOL_MODEL = "llama3-groq-8b-8192-tool-use-preview"
LLAMA_405_MODEL = "llama3.1-405b-reasoning"
LLAMA_70B_MODEL = "llama-3.1-70b-versatile"
LLAMA_8B_INSTANT = "llama-3.1-8b-instant"

GROQ_MODELS = [
    LLAMA_8_MODEL,
    LLAMA_8_TOOL_MODEL,
    LLAMA_405_MODEL,
    LLAMA_70B_MODEL,
    LLAMA_8B_INSTANT,
    MIXTRAL_MODEL
]

OPENAI_MODELS = [
    GPT_35_TURBO_MODEL,
    GPT_4O_MODEL,
    GPT_4O_MINI,
    O1_MINI_MODEL
]

# Text chunking constants
DEFAULT_CHUNK_SIZE = 250
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TEXT_PREVIEW_LENGTH = 200
DEFAULT_MAX_CONCURRENCY = 5

# PDF processing constants
DEFAULT_SUMMARY_SYSTEM_PROMPT = """
You are a professional summarizer. Your task is to create a concise and comprehensive summary 
of the provided text. Focus on capturing the main ideas, key concepts, and important details.
The summary should be approximately 15% of the original text length, but no more than 1000 words.
"""

DEFAULT_SUMMARY_USER_TEMPLATE = """
Please summarize the following text:

{text}
"""

# Knowledge graph constants
KG_SYSTEM_PROMPT = """
- You are a top-tier algorithm designed for extracting information in structured formats to build a concise and meaningful knowledge graph.
- Your task is to identify the most important concepts and entities in the text and the relations between them.
- You will provide descriptions for each node as they would appear on a flashcard.
- You will use the summary of the text provided to guide which concepts and entities are most important to extract.
- You should use the summary to correct any typos in the source text based on the context provided.
- You will always output node ids in all lowercase with spaces between words.

# Output Format #
You will output the knowledge graph in the following format, it is extremely important that you follow this format:
nodes: A list of nodes, where each node is a dictionary with the following keys:
    id: The unique identifier of the node. Must be all lowercase with spaces between words.
    description: The description of the node as would be read on a flashcard.
relationships: A list of relationships, where a relationship is a dictionary with the following keys:
    source: The unique identifier of the source node, must match a node in the nodes list. Must be all lowercase with spaces between words.
    target: The unique identifier of the target node, must match a node in the nodes list. Must be all lowercase with spaces between words.
    type: The type of the relationship.

## IMPORTANT GUIDELINES ##
- Focus on extracting the most significant entities and thoroughly identify meaningful relationships between them to create a well-connected graph.
- Ensure that all important nodes are interconnected through relevant relationships where appropriate.
- Maintain Entity Consistency: When extracting entities or concepts, it's vital to ensure consistency.
- If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"), always use the most complete identifier for that entity.

## FINAL POINT ##
It is important that you focus on the most important nodes and establish as many meaningful relationships as possible to build a concise and interconnected knowledge graph.
"""

KG_USER_TEMPLATE = """
Based on the following text and summary, extract the most important entities/concepts and identify as many meaningful relationships between them as possible.
Please remember to provide a description for each node as it would appear on a flashcard.

Summary of document:
{summary}

Text to extract from:
{text}
"""

# Entity merging constants
ENTITY_SIMILARITY_THRESHOLD = 0.85
ENTITY_MERGING_SYSTEM_PROMPT = """You are an expert in knowledge graph entity resolution. Your task is to analyze groups of similar entities and decide:
1. Whether they should be merged into a single entity
2. What the final entity ID (name) should be
3. What the final entity description should be

For each group, consider:
- Semantic similarity of the entities
- Whether they refer to the same real-world concept
- Which name is most canonical, clear, and descriptive
- Which description is most comprehensive and accurate

Be more aggressive in merging entities that likely refer to the same concept, even if their names differ somewhat.
For example, "personal portfolio website" and "personal website" should be merged as they refer to the same concept.

Return your decisions in a structured format. For each group, decide if they should be merged and provide the final entity details.
"""

ENTITY_MERGING_USER_TEMPLATE = """I have identified the following groups of potentially similar entities in a knowledge graph:

{entity_groups}

For each group, please decide:
1. Should these entities be merged? (true/false)
2. If yes, what should be the final entity ID (name)?
3. If yes, what should be the final entity description?

Be more aggressive in merging entities that likely refer to the same concept, even if their names differ somewhat.
For example, "personal portfolio website" and "personal website" should be merged as they refer to the same concept.

Return your analysis in the following JSON format:
```json
[
  {
    "group_id": 0,
    "should_merge": true/false,
    "final_id": "chosen entity name",
    "final_description": "chosen entity description",
    "indices": [list of indices to merge]
  },
  ...
]
```

Only include groups where should_merge is true in your response.
"""

# Stop words for entity merging
ENTITY_STOP_WORDS = {'a', 'an', 'the', 'and', 'or', 'but', 'of', 'for', 'with', 'in', 'on', 'at', 'to', 'from'}
MIN_WORD_LENGTH_FOR_PREFIX_CHECK = 4