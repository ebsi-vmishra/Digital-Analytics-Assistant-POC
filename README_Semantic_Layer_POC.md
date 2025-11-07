# Semantic Layer POC — Enhancing Vanna’s Training and Accuracy

## 1. Overview
Vanna is an NL→SQL generation engine that can already introspect a database’s `information_schema` to understand structure (tables, columns, and types).  
However, that raw structural view **lacks business meaning** — it knows *what exists* but not *what it means*.

This Semantic Layer POC bridges that gap. It builds a multi-layer, reusable metadata foundation that provides:
- Business context,
- Consistent vocabulary,
- Data normalization, and
- Join logic awareness.

The result is a **Vanna training bundle** that improves both **recall** (understanding natural language phrasing) and **precision** (building correct SQL).

---

## 2. How Vanna Trains (Base Capability)
By default, Vanna can:
- Read **schema metadata** via `information_schema` (table names, column names, data types, foreign keys),
- Use **example questions & SQL** pairs to learn phrasing patterns,
- Embed **documentation** and **descriptions** for retrieval,
- Perform **semantic search** over metadata chunks,
- Generate SQL via **LLM reasoning** based on retrieved chunks.

However, this default setup is **flat**:
- It knows syntax, not semantics.
- It lacks cross-table relationships beyond FK hints.
- It cannot disambiguate business synonyms (e.g., *smoker* vs. *tobacco user*).
- It can’t normalize enums or flags (e.g., *FT/PT* → *Full-Time/Part-Time*).
- It cannot unify multiple client schemas under one conceptual model.

That’s where the Semantic Layer POC adds value.

---

## 3. The POC Architecture
### High-level flow
```
SQL Server → Ingestion → Profiling → Documentation → Concepts → Synonyms → Vanna Bundle
```

---

## 4. Layer-by-Layer: Purpose, Value, and Impact on Vanna
| Layer | Core Purpose | Value to Semantic Layer | How It Improves Vanna Beyond `information_schema` |
|--------|---------------|-------------------------|----------------------------------------------------|
| Ingestion | Extracts structure (tables, columns, FKs) | Baseline for profiling & docs | Ensures accuracy and alignment |
| Profiling | Data stats (enums, PKs, examples) | Adds quant context | Improves flag/enum understanding |
| Documentation | LLM-based business meaning | Translates schema to natural language | Improves retrieval and comprehension |
| Concept Layer | Normalized business entities | Defines "business truth" | Enables cross-tenant consistency |
| Synonyms | Alias → Target map | Expands vocabulary | Improves NL recall and precision |
| Vanna Bundle | Export of all artifacts | Training package | Unified retrieval and faster onboarding |

---

## 5. Flow of Semantic Layer Creation
1. **Ingest Schema** → schema.json, relationships.json  
2. **Profile Data** → profiles/*.json, profiles_summary.csv  
3. **Auto Documentation (LLM)** → schema_docs.jsonl  
4. **Concept Layer (LLM)** → concept_*.csv  
5. **Synonyms (LLM)** → synonyms.json, attribute_map.json  
6. **Export Bundle** → vanna_bundle/*

---

## 6. Holistic Value: How the Semantic Layer Improves Vanna
| Dimension | Vanna Base | With Semantic Layer |
|------------|-------------|----------------------|
| Schema Awareness | Structural | Structural + Semantic |
| Join Reasoning | FK-only | FK + Profile grain |
| Flag Handling | Raw | Normalized via concepts |
| NL Understanding | Name-based | Docs + Synonyms |
| Cross-Tenant | None | Concept normalization |
| Vocabulary | Technical | Business language |
| Accuracy | Prompt-dependent | Context-rich |
| Training Speed | Manual | Automated, reusable |

---

## 7. Which Layers Matter Most
| Layer | Importance | Why |
|--------|-------------|-----|
| Documentation | ⭐⭐⭐⭐ | Enables retrieval alignment |
| Concepts | ⭐⭐⭐⭐ | Unifies meaning |
| Synonyms | ⭐⭐⭐⭐ | Expands vocabulary |
| Profiling | ⭐⭐⭐ | Feeds others |
| Ingestion | ⭐⭐ | Structural baseline |
| Bundle | ⭐⭐⭐⭐ | Integration artifact |

---

## 8. Semantic Layer Flow
```
Ingest → Profile → Document → Concepts → Synonyms → Export → Train Vanna
```

---

## 9. Final Takeaway
Without Semantic Layer: Vanna knows table and column names.  
With Semantic Layer: Vanna understands meaning, relationships, and user phrasing — enabling contextually correct, business-aligned SQL.

---

## 10. Summary: Why This Matters
| Benefit | Enabled By | Impact |
|----------|-------------|--------|
| Human-level understanding | Docs + Synonyms | Better NL parsing |
| Cross-tenant consistency | Concepts | Unified vocabulary |
| Normalized filters | Profiling + Concepts | Correct predicates |
| Accurate joins | Relationships + Profiles | Correct SQL joins |
| Rapid onboarding | Bundle Export | Faster deployment |
| Reduced hallucination | Context grounding | Trustworthy SQL |
