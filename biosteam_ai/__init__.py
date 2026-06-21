"""BioSTEAM AI Assistant.

A validated natural-language interface over a curated set of BioSTEAM
biorefinery models. The assistant does not generate free-form simulation
code; instead an LLM drives a typed, allowlisted tool layer that loads
models, changes vetted parameters, runs the simulation, and reads back
techno-economic metrics with full provenance.
"""

__version__ = "0.1.0"
