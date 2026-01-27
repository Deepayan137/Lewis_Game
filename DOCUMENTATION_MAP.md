# Documentation Map

Visual guide to all project documentation and how they connect.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         LEWIS GAME DOCUMENTATION                             │
└─────────────────────────────────────────────────────────────────────────────┘

                                  START HERE
                                      ↓
                          ┌───────────────────────┐
                          │   README.md           │ ← Main project overview
                          │   (You are here)      │   All pipelines & quick start
                          └───────────┬───────────┘
                                      │
                 ┌────────────────────┼────────────────────┐
                 │                    │                    │
                 ↓                    ↓                    ↓
      ┌──────────────────┐ ┌──────────────────┐ ┌──────────────────┐
      │ QUICK_REFERENCE  │ │ Data Preparation │ │ Training         │
      │                  │ │ Pipeline         │ │ Pipeline         │
      │ Fast lookup for  │ │                  │ │                  │
      │ common commands  │ │ src/data_prepare/│ │ train_src/       │
      │ and configs      │ │ README.md        │ │ README.md        │
      └──────────────────┘ └────────┬─────────┘ └────────┬─────────┘
                                     │                    │
                                     │                    │
                                     ↓                    ↓
                          ┌─────────────────────┐ ┌─────────────────────┐
                          │ Inference &         │ │ Services            │
                          │ Evaluation          │ │ • listener_service  │
                          │                     │ │ • speaker_service   │
                          │ src/README.md       │ │                     │
                          └─────────┬───────────┘ │ Utilities           │
                                    │             │ • dist_helpers      │
                                    │             │ • logger            │
                                    │             └─────────────────────┘
                                    ↓
                          ┌─────────────────────┐
                          │ Description         │
                          │ Processing          │
                          │                     │
                          │ eval_utils/         │
                          │ description_        │
                          │ processing/         │
                          │ README.md           │
                          └─────────────────────┘

                            TECHNICAL DOCS
                                  ↓
                 ┌────────────────┴────────────────┐
                 │                                 │
                 ↓                                 ↓
      ┌──────────────────────┐         ┌──────────────────────┐
      │ REFACTORING_SUMMARY  │         │ IMPORT_FIX_SUMMARY   │
      │                      │         │                      │
      │ Description          │         │ PEP 8 import order   │
      │ processing module    │         │ compliance fixes     │
      │ reorganization       │         │ (train_src/)         │
      └──────────────────────┘         └──────────────────────┘
```

---

## Documentation Hierarchy

### Level 1: Entry Points (Start Here)

#### 📘 [README.md](README.md) - **Main Documentation**
- **Audience**: Everyone
- **Purpose**: Complete project overview
- **Content**:
  - Quick start guide
  - All pipeline overviews
  - Installation instructions
  - Common workflows
  - Troubleshooting
- **When to use**: First time setup, general reference

#### 📘 [QUICK_REFERENCE.md](QUICK_REFERENCE.md) - **Cheat Sheet**
- **Audience**: Developers (daily use)
- **Purpose**: Fast command lookup
- **Content**:
  - Common commands
  - Configuration snippets
  - Quick fixes
  - File locations
- **When to use**: During development, debugging

---

### Level 2: Pipeline-Specific Documentation

#### 📗 [src/data_prepare/README.md](src/data_prepare/README.md)
- **Topic**: Data Preparation Pipeline
- **Audience**: Data engineers, researchers preparing datasets
- **Content**:
  - Catalog creation (Step 1)
  - Concept splits (Step 2)
  - CLIP retrieval (Steps 3-4)
  - HuggingFace conversion (Step 5)
- **Prerequisites**: Raw image data
- **Next steps**: → Training or Inference

#### 📗 [train_src/README.md](train_src/README.md) ⭐ **Most Comprehensive**
- **Topic**: Training Pipeline
- **Audience**: ML engineers, researchers
- **Content**:
  - Complete training setup
  - Listener/Speaker services
  - GRPO training details
  - Reward functions
  - Configuration guide
  - Debugging & optimization
- **Prerequisites**: Data catalog + processed dataset
- **Next steps**: → Trained models for inference

#### 📗 [src/README.md](src/README.md)
- **Topic**: Inference & Evaluation Pipeline
- **Audience**: Researchers, evaluators
- **Content**:
  - Description generation
  - Personalization task
  - Recognition task
  - VQA task
  - Model types
- **Prerequisites**: Data catalog + trained models
- **Next steps**: → Results analysis

#### 📗 [src/eval_utils/description_processing/README.md](src/eval_utils/description_processing/README.md)
- **Topic**: Description Processing Module
- **Audience**: Developers working on description quality
- **Content**:
  - Module structure
  - Evaluation (detect state/location)
  - Refinement (remove attributes)
  - Usage examples
- **Prerequisites**: Generated descriptions
- **Next steps**: → Refined descriptions for evaluation

---

### Level 3: Technical Documentation

#### 📙 [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)
- **Topic**: Description Processing Refactoring
- **Audience**: Developers, maintainers
- **Content**:
  - Before/after comparison
  - Module structure rationale
  - Migration guide
  - Benefits
- **When to read**: Understanding the refactored code structure

#### 📙 [IMPORT_FIX_SUMMARY.md](IMPORT_FIX_SUMMARY.md)
- **Topic**: Import Order Fixes (PEP 8 Compliance)
- **Audience**: Developers, contributors
- **Content**:
  - PEP 8 import standards
  - Before/after for each file
  - Common issues fixed
  - Recommendations
- **When to read**: Contributing code, understanding import structure

---

## Reading Order by Use Case

### Use Case 1: New to the Project
```
1. README.md (Main overview)
   ↓
2. QUICK_REFERENCE.md (Commands & configs)
   ↓
3. Pick your pipeline:
   - Data prep → src/data_prepare/README.md
   - Training → train_src/README.md
   - Evaluation → src/README.md
```

### Use Case 2: Running Experiments
```
1. QUICK_REFERENCE.md (Fast command lookup)
   ↓
2. src/README.md (Evaluation tasks)
   ↓
3. train_src/README.md (If training new models)
```

### Use Case 3: Setting Up Training
```
1. README.md (Installation & setup)
   ↓
2. src/data_prepare/README.md (Prepare data)
   ↓
3. train_src/README.md (Training guide) ⭐
   ↓
4. QUICK_REFERENCE.md (Configs & commands)
```

### Use Case 4: Contributing Code
```
1. README.md (Project structure)
   ↓
2. IMPORT_FIX_SUMMARY.md (Code style)
   ↓
3. REFACTORING_SUMMARY.md (Module design)
   ↓
4. Relevant pipeline README
```

### Use Case 5: Processing Descriptions
```
1. README.md (Overview)
   ↓
2. src/README.md (Generate descriptions)
   ↓
3. eval_utils/description_processing/README.md
```

---

## Content Comparison

| Document | Length | Depth | Audience | Update Frequency |
|----------|--------|-------|----------|------------------|
| **README.md** | Long | Medium | Everyone | Medium |
| **QUICK_REFERENCE.md** | Short | Shallow | Developers | High |
| **train_src/README.md** | Very Long | Deep | ML Engineers | Low |
| **src/README.md** | Long | Medium | Researchers | Low |
| **data_prepare/README.md** | Medium | Medium | Data Engineers | Low |
| **description_processing/README.md** | Medium | Deep | Developers | Low |
| **REFACTORING_SUMMARY.md** | Long | Deep | Maintainers | None (historical) |
| **IMPORT_FIX_SUMMARY.md** | Long | Deep | Contributors | None (historical) |

---

## Cross-References

### From Main README
- → Data Preparation: `src/data_prepare/README.md`
- → Training: `train_src/README.md`
- → Evaluation: `src/README.md`
- → Description Processing: `src/eval_utils/description_processing/README.md`
- → Quick Ref: `QUICK_REFERENCE.md`

### From Training README
- ← Main: `README.md`
- → Data Prep: `src/data_prepare/README.md` (for dataset creation)
- → Evaluation: `src/README.md` (to evaluate trained models)

### From Inference README
- ← Main: `README.md`
- → Data Prep: `src/data_prepare/README.md` (prerequisites)
- → Description Processing: `src/eval_utils/description_processing/README.md`

### From Description Processing README
- ← Main: `README.md`
- ← Inference: `src/README.md` (generate descriptions first)
- → Refactoring: `REFACTORING_SUMMARY.md` (design rationale)

---

## Search Index

### By Topic

**Installation & Setup**
- `README.md` - Installation section
- `QUICK_REFERENCE.md` - Setup commands

**Data Preparation**
- `src/data_prepare/README.md` - Complete data pipeline
- `README.md` - Data preparation overview

**Training**
- `train_src/README.md` - Comprehensive training guide ⭐
- `README.md` - Training overview
- `QUICK_REFERENCE.md` - Training commands

**Evaluation**
- `src/README.md` - All evaluation tasks
- `README.md` - Evaluation overview

**Description Processing**
- `src/eval_utils/description_processing/README.md` - Module guide
- `REFACTORING_SUMMARY.md` - Refactoring details

**Troubleshooting**
- `README.md` - Common issues
- `QUICK_REFERENCE.md` - Quick fixes
- `train_src/README.md` - Training-specific debugging

**Configuration**
- `README.md` - Environment setup
- `QUICK_REFERENCE.md` - Config snippets
- `train_src/README.md` - Training configs

**Code Style**
- `IMPORT_FIX_SUMMARY.md` - Import order standards
- `REFACTORING_SUMMARY.md` - Module organization

---

## Update Guidelines

### When to Update Each Document

**README.md** (Main)
- New features or pipelines added
- Major configuration changes
- Installation requirements change
- Project structure changes

**QUICK_REFERENCE.md**
- New common commands discovered
- Configuration defaults change
- New quick fixes identified

**Pipeline READMEs** (train_src/, src/, etc.)
- API changes in respective pipeline
- New scripts added
- Configuration options change
- Output formats change

**Technical Docs** (REFACTORING_SUMMARY, IMPORT_FIX_SUMMARY)
- Generally static (historical records)
- Update only if major refactoring occurs

---

## Maintenance Checklist

### Monthly Review
- [ ] Check all commands in QUICK_REFERENCE.md still work
- [ ] Verify installation instructions in README.md
- [ ] Test all quick start commands
- [ ] Update "Last Updated" dates

### After Major Changes
- [ ] Update relevant README files
- [ ] Add new commands to QUICK_REFERENCE.md
- [ ] Update cross-references
- [ ] Create new technical summary if needed (like REFACTORING_SUMMARY.md)

### Before Release
- [ ] All READMEs reviewed
- [ ] All examples tested
- [ ] Cross-references valid
- [ ] No broken links

---

## Quick Navigation

**I want to...**

- **Get started** → [README.md](README.md)
- **Find a command quickly** → [QUICK_REFERENCE.md](QUICK_REFERENCE.md)
- **Prepare my data** → [src/data_prepare/README.md](src/data_prepare/README.md)
- **Train a model** → [train_src/README.md](train_src/README.md) ⭐
- **Evaluate a model** → [src/README.md](src/README.md)
- **Process descriptions** → [src/eval_utils/description_processing/README.md](src/eval_utils/description_processing/README.md)
- **Understand the refactoring** → [REFACTORING_SUMMARY.md](REFACTORING_SUMMARY.md)
- **Follow code style** → [IMPORT_FIX_SUMMARY.md](IMPORT_FIX_SUMMARY.md)
- **See this map** → [DOCUMENTATION_MAP.md](DOCUMENTATION_MAP.md) (you are here)

---

**Last Updated**: January 27, 2026

This map will be updated as documentation evolves. If you add new documentation, please update this map!
