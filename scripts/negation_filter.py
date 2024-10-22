# scripts/negation_filter.py

import spacy
from negspacy.negation import Negex
from negspacy.termsets import termset
from itertools import product
from functools import lru_cache

# Load the clinical SpaCy model
try:
    nlp = spacy.load("en_core_sci_md")
except OSError:
    raise ValueError("SpaCy model 'en_core_sci_md' not found. Please make sure it's installed.")

# Instantiate the termset object
ts = termset("en_clinical")  # Instantiate the English termset

# Initialize Negex with the required parameters
negex = Negex(
    nlp,
    name="negex",
    neg_termset=ts.get_patterns(),  # Access negation terms as an attribute
    ent_types=None,  # Set to None to apply negation detection to all entity types
    extension_name="negex",
    chunk_prefix=["no", "without", "absence", "negative for"]
)

# Add Negex to the pipeline if not already present
if "negex" not in nlp.pipe_names:
    nlp.add_pipe("negex")

modifiers = ["cardiac"]

# Function to automatically generate disease terms with modifiers
@lru_cache(maxsize=None)
def expand_disease_terms(base_terms_tuple, modifiers_tuple):
    base_terms = list(base_terms_tuple)
    modifiers = list(modifiers_tuple)
    # Create combinations of modifiers and base terms
    expanded_terms = set(base_terms)  # Add base terms initially
    for modifier, base in product(modifiers, base_terms):
        expanded_terms.add(f"{modifier} {base}")
    return list(expanded_terms)


def negation_filter(text, disease_terms):
    # Use caching for expanded disease terms
    expanded_disease_terms = expand_disease_terms(tuple(disease_terms), tuple(modifiers))
    doc = nlp(text)
    for ent in doc.ents:
        # Check if the entity text matches any of the disease terms
        if ent.text.lower() in expanded_disease_terms:
            if ent._.negex:
                return True  # Negative finding detected
    return False  # Not negative (could be positive or neutral)

