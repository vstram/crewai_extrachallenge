# Temperature Configuration Guide

## Overview

**Question**: "Now which value of temperature will be used? Is there a way to configure the temperature in a non-code file, such as .env?"

**Answer**: Temperature is now configured in `config/agents.yaml` for each agent. The model uses `${MODEL}` variable from `.env`, allowing you to change models without editing code.

---

## Current Configuration

### Temperature Settings by Agent

All temperatures are configured in `src/crewai_extrachallenge/config/agents.yaml`:

```yaml
data_analyst:
  llm:
    model: ${MODEL}      # Uses MODEL from .env
    temperature: 0.1     # Low - precise, deterministic analysis

pattern_recognition_agent:
  llm:
    model: ${MODEL}      # Uses MODEL from .env
    temperature: 0.3     # Medium - creative pattern discovery

classification_agent:
  llm:
    model: ${MODEL}      # Uses MODEL from .env
    temperature: 0.1     # Low - consistent classification

reporting_analyst:
  llm:
    model: ${MODEL}      # Uses MODEL from .env
    temperature: 0.2     # Slight creativity - clear explanations
```

### Temperature Rationale

| Agent | Temperature | Reason |
|-------|-------------|--------|
| **data_analyst** | **0.1** | Needs **precise, deterministic** statistical analysis. Low temperature ensures consistent tool calls and accurate calculations. |
| **pattern_recognition_agent** | **0.3** | Needs **creative pattern discovery**. Medium temperature allows exploring different fraud indicators while staying focused. |
| **classification_agent** | **0.1** | Needs **consistent classification** decisions. Low temperature ensures reliable fraud vs legitimate labeling. |
| **reporting_analyst** | **0.2** | Needs **clear, engaging writing**. Slight creativity helps with explanations while maintaining accuracy. |

---

## How Configuration Works

### 1. Model from .env

The `${MODEL}` variable references the `MODEL` setting in `.env`:

**.env**:
```env
MODEL=gpt-4-turbo-preview
```

**agents.yaml**:
```yaml
data_analyst:
  llm:
    model: ${MODEL}  # Resolves to "gpt-4-turbo-preview"
    temperature: 0.1
```

### 2. Temperature in YAML

Each agent specifies its own temperature in `agents.yaml`:

- **Not in .env** (would apply same temperature to all agents)
- **In agents.yaml** (allows different temperatures per agent)

---

## Temperature Scale Reference

### Temperature: 0.0 - 0.2 (Deterministic)
- **Use for**: Mathematical calculations, data analysis, classification
- **Behavior**: Highly consistent, predictable outputs
- **Agents**: `data_analyst` (0.1), `classification_agent` (0.1)

### Temperature: 0.2 - 0.4 (Balanced)
- **Use for**: Report writing, pattern recognition
- **Behavior**: Mix of consistency and creativity
- **Agents**: `reporting_analyst` (0.2), `pattern_recognition_agent` (0.3)

### Temperature: 0.5 - 0.7 (Creative)
- **Use for**: Brainstorming, exploratory analysis
- **Behavior**: More varied, creative outputs
- **Agents**: None (not recommended for fraud detection)

### Temperature: 0.8 - 1.0 (Highly Creative)
- **Use for**: Creative writing, idea generation
- **Behavior**: Unpredictable, diverse outputs
- **Agents**: None (not suitable for fraud detection)

---

## Recommended Temperatures by Use Case

### For Maximum Accuracy (Production)
```yaml
data_analyst:
  temperature: 0.05  # Even more deterministic

pattern_recognition_agent:
  temperature: 0.15  # Lower for consistency

classification_agent:
  temperature: 0.05  # Even more deterministic

reporting_analyst:
  temperature: 0.15  # Lower for formal reports
```

### For Exploratory Analysis (Research)
```yaml
data_analyst:
  temperature: 0.2   # Slightly more flexible

pattern_recognition_agent:
  temperature: 0.4   # More creative pattern discovery

classification_agent:
  temperature: 0.2   # Slightly more flexible

reporting_analyst:
  temperature: 0.3   # More engaging narratives
```

### For Testing/Development (Current Settings)
```yaml
data_analyst:
  temperature: 0.1   # Good balance

pattern_recognition_agent:
  temperature: 0.3   # Good balance

classification_agent:
  temperature: 0.1   # Good balance

reporting_analyst:
  temperature: 0.2   # Good balance
```

---

## How to Change Temperature

### Method 1: Edit agents.yaml (Recommended)

**File**: `src/crewai_extrachallenge/config/agents.yaml`

```yaml
data_analyst:
  llm:
    model: ${MODEL}
    temperature: 0.1  # ← Change this value

pattern_recognition_agent:
  llm:
    model: ${MODEL}
    temperature: 0.3  # ← Change this value
```

**Advantages**:
- ✅ Different temperatures per agent
- ✅ Well-documented in config file
- ✅ No code changes needed

### Method 2: Environment Variables (Advanced)

You can also use environment variables for temperatures:

**.env**:
```env
MODEL=gpt-4-turbo-preview
TEMP_DATA_ANALYST=0.1
TEMP_PATTERN_AGENT=0.3
TEMP_CLASSIFICATION=0.1
TEMP_REPORTING=0.2
```

**agents.yaml**:
```yaml
data_analyst:
  llm:
    model: ${MODEL}
    temperature: ${TEMP_DATA_ANALYST}

pattern_recognition_agent:
  llm:
    model: ${MODEL}
    temperature: ${TEMP_PATTERN_AGENT}
```

**Advantages**:
- ✅ Change all temperatures without editing YAML
- ✅ Easy A/B testing
- ✅ Environment-specific settings (dev/prod)

---

## Testing Different Temperatures

### Experiment Protocol

1. **Baseline (Current)**:
   ```yaml
   data_analyst: 0.1
   pattern_recognition_agent: 0.3
   classification_agent: 0.1
   reporting_analyst: 0.2
   ```

2. **Conservative (More Deterministic)**:
   ```yaml
   data_analyst: 0.05
   pattern_recognition_agent: 0.15
   classification_agent: 0.05
   reporting_analyst: 0.1
   ```

3. **Creative (More Exploratory)**:
   ```yaml
   data_analyst: 0.2
   pattern_recognition_agent: 0.4
   classification_agent: 0.2
   reporting_analyst: 0.3
   ```

### Test Script

```bash
#!/bin/bash

# Test different temperature configurations
temperatures=("0.05 0.15 0.05 0.1" "0.1 0.3 0.1 0.2" "0.2 0.4 0.2 0.3")
names=("conservative" "baseline" "creative")

for i in {0..2}; do
    echo "Testing ${names[$i]} temperatures..."

    # Update agents.yaml (manual step or use sed)
    # Run analysis
    crewai run > "results_${names[$i]}.log" 2>&1

    # Compare results
    echo "Results saved to results_${names[$i]}.log"
done
```

### Metrics to Compare

- **Consistency**: Run 3 times, check if outputs are similar
- **Tool Call Success Rate**: % of successful tool executions
- **Report Quality**: Clarity, completeness, accuracy
- **Execution Time**: Time to complete analysis

---

## Default Behavior (No LLM Config)

**Before** (when we removed hardcoded LLMs):
```python
# In crew.py - no LLM specified
return Agent(
    config=self.agents_config['data_analyst'],
    tools=[...],
    verbose=True
)
```

**What happens**:
- CrewAI uses `MODEL` from `.env`
- **Default temperature**: Varies by model
  - OpenAI models: `temperature=0.7` (default)
  - Ollama models: `temperature=0.8` (default)

**Problem**: Too high for fraud detection (inconsistent results)

**Solution**: Explicitly set temperature in `agents.yaml` ✅

---

## Model-Specific Temperature Considerations

### OpenAI Models (GPT-4, GPT-3.5)
- **Range**: 0.0 - 2.0
- **Recommended**: 0.0 - 0.5 for fraud detection
- **Default**: 0.7 (if not specified)

```yaml
llm:
  model: ${MODEL}  # gpt-4-turbo-preview
  temperature: 0.1  # Low for accuracy
```

### Ollama Models (llama3.1, mistral)
- **Range**: 0.0 - 1.0
- **Recommended**: 0.0 - 0.4 for fraud detection
- **Default**: 0.8 (if not specified)

```yaml
llm:
  model: ${MODEL}  # ollama/llama3.1:8b
  temperature: 0.1  # Low for accuracy
```

### Claude Models (if supported)
- **Range**: 0.0 - 1.0
- **Recommended**: 0.0 - 0.3 for fraud detection
- **Default**: 1.0 (if not specified)

---

## Common Issues and Solutions

### Issue 1: Results Too Random
**Symptom**: Different outputs each run, inconsistent tool calls

**Solution**: **Lower temperature**
```yaml
data_analyst:
  temperature: 0.05  # Was 0.1, reduce to 0.05
```

### Issue 2: Results Too Repetitive
**Symptom**: Same patterns found every time, missing edge cases

**Solution**: **Increase temperature slightly**
```yaml
pattern_recognition_agent:
  temperature: 0.4  # Was 0.3, increase to 0.4
```

### Issue 3: Tool Calls Failing
**Symptom**: Agent not calling tools with correct parameters

**Solution**: **Lower temperature** (especially for data_analyst)
```yaml
data_analyst:
  temperature: 0.0  # Absolute determinism
```

---

## Best Practices

### 1. Start Low, Increase Gradually
- Begin with `temperature: 0.1`
- If too rigid, increase by 0.1 increments
- Test after each change

### 2. Different Temperatures for Different Roles
- **Analysis agents**: Low (0.0 - 0.2)
- **Pattern agents**: Medium (0.2 - 0.4)
- **Reporting agents**: Low-medium (0.1 - 0.3)

### 3. Document Your Changes
```yaml
data_analyst:
  llm:
    temperature: 0.15  # Increased from 0.1 for better pattern recognition
```

### 4. Use Environment Variables for A/B Testing
```env
# Production
TEMP_DATA_ANALYST=0.05

# Development
TEMP_DATA_ANALYST=0.2
```

---

## Summary

### Current Setup ✅

| Configuration | Value | Location |
|---------------|-------|----------|
| **Model** | From `.env` `MODEL` | `agents.yaml` uses `${MODEL}` |
| **data_analyst temp** | `0.1` | `config/agents.yaml` line 10 |
| **pattern_recognition temp** | `0.3` | `config/agents.yaml` line 21 |
| **classification temp** | `0.1` | `config/agents.yaml` line 32 |
| **reporting temp** | `0.2` | `config/agents.yaml` line 43 |

### How to Change

1. **Edit** `src/crewai_extrachallenge/config/agents.yaml`
2. **Modify** temperature values (0.0 - 1.0)
3. **Save** and run `crewai run`
4. **No code changes** needed ✅

### Recommended Settings

**For production** (maximum reliability):
- data_analyst: `0.05`
- pattern_recognition: `0.15`
- classification: `0.05`
- reporting: `0.15`

**For development** (current, balanced):
- data_analyst: `0.1` ✅
- pattern_recognition: `0.3` ✅
- classification: `0.1` ✅
- reporting: `0.2` ✅

**For research** (exploratory):
- data_analyst: `0.2`
- pattern_recognition: `0.4`
- classification: `0.2`
- reporting: `0.3`

---

## Files Modified

| File | Change | Lines |
|------|--------|-------|
| `config/agents.yaml` | Added `llm.temperature: 0.1` to `data_analyst` | 8-10 |
| `config/agents.yaml` | Added `llm.temperature: 0.3` to `pattern_recognition_agent` | 19-21 |
| `config/agents.yaml` | Added `llm.temperature: 0.1` to `classification_agent` | 30-32 |
| `config/agents.yaml` | Added `llm.temperature: 0.2` to `reporting_analyst` | 41-43 |
| `TEMPERATURE_CONFIGURATION_GUIDE.md` | Created | - |

---

## Quick Reference

```bash
# View current temperatures
cat src/crewai_extrachallenge/config/agents.yaml | grep -A 2 "llm:"

# Change temperature for data_analyst
# Edit: src/crewai_extrachallenge/config/agents.yaml
# Line 10: temperature: 0.1  # Change this

# Test new configuration
crewai run

# Compare results with different temperatures
diff results_temp_01.log results_temp_03.log
```

**Temperature is now fully configurable via `agents.yaml` without any code changes!** 🎉
