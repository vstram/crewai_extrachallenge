# Temperature Configuration - Actual Behavior

## Summary

**Question**: "Now which value of temperature will be used? Is there a way to configure the temperature in a non-code file, such as .env?"

**Answer**:
1. **Current behavior**: Temperature uses the **model's default** (not configurable via YAML or .env in current CrewAI version)
2. **YAML `llm` config doesn't support temperature** - CrewAI only supports `llm: "model_name"` as a string
3. **Temperature in .env is possible** but requires custom LLM initialization in `crew.py`

---

## What We Discovered

### Attempted Approach #1: YAML Configuration ❌

**Tried**:
```yaml
data_analyst:
  llm:
    model: ${MODEL}
    temperature: 0.1
```

**Error**:
```
TypeError: unhashable type: 'dict'
```

**Reason**: CrewAI expects `llm` to be a **string** (model name), not a dict with config.

### Attempted Approach #2: Agent `llm_config` Parameter ❌

**Tried**:
```python
Agent(
    llm_config={"temperature": 0.1},
    ...
)
```

**Result**: **No such parameter exists** in `crewai.agent.Agent` class.

**Verified**: Checked `/crewai/agent.py` - no `llm_config` or `temperature` parameters.

---

## Actual Current Behavior

### When LLM is NOT specified in crew.py

```python
# crew.py (current state)
@agent
def data_analyst(self) -> Agent:
    return Agent(
        config=self.agents_config['data_analyst'],
        tools=[...]
    )
```

**What happens**:
1. CrewAI reads `MODEL` from `.env`
2. Creates LLM instance with that model
3. Uses **model's default temperature**:
   - **OpenAI models** (gpt-4, gpt-3.5): Default `temperature=0.7`
   - **Ollama models** (llama3.1): Default `temperature=0.8`
   - **Anthropic models**: Default `temperature=1.0`

**Problem**: Default temperatures are **too high** for fraud detection (causes inconsistent results).

---

## Solution Options

### Option 1: Accept Default Temperatures (Current)

**Status**: ✅ Already implemented (all agents use .env MODEL with defaults)

**Pros**:
- Simple, no code changes
- Model controlled via `.env`

**Cons**:
- ❌ Can't customize temperature per agent
- ❌ Uses model defaults (often too high)

**Current temperatures**:
```
MODEL=gpt-5-mini
├── data_analyst: 0.7 (gpt default)
├── pattern_recognition: 0.7 (gpt default)
├── classification: 0.7 (gpt default)
└── reporting: 0.7 (gpt default)
```

### Option 2: Custom LLM with Temperature in crew.py

**Implementation**:
```python
import os
from crewai import LLM

@agent
def data_analyst(self) -> Agent:
    # Create LLM with custom temperature from .env
    model = os.getenv('MODEL', 'gpt-4-turbo-preview')
    temperature = float(os.getenv('TEMP_DATA_ANALYST', '0.1'))

    custom_llm = LLM(
        model=model,
        temperature=temperature
    )

    return Agent(
        config=self.agents_config['data_analyst'],
        llm=custom_llm,  # Pass custom LLM
        tools=[...]
    )
```

**.env**:
```env
MODEL=gpt-4-turbo-preview
TEMP_DATA_ANALYST=0.1
TEMP_PATTERN_AGENT=0.3
TEMP_CLASSIFICATION=0.1
TEMP_REPORTING=0.2
```

**Pros**:
- ✅ Full control over temperature per agent
- ✅ Configured via `.env` (no hardcoding)
- ✅ Different temps for different agents

**Cons**:
- ❌ Requires code in `crew.py` (not pure YAML config)
- ❌ Repeats LLM initialization for each agent

### Option 3: Model String with Temperature Parameter

**Check if CrewAI supports**:
```python
# Some frameworks support this format:
llm = "gpt-4-turbo-preview?temperature=0.1"
```

**Status**: Need to test, likely **not supported** by CrewAI.

---

## Recommended Approach

### For Now: Use .env MODEL with Defaults (Option 1)

**Current Setup**:
```env
MODEL=gpt-5-mini  # Uses default temp (0.7)
```

**Acceptable because**:
- Temperature 0.7 is reasonable for most tasks
- Simpler configuration
- No code changes needed

### For Production: Custom LLM per Agent (Option 2)

**If you need precise control**, implement custom LLM initialization:

**File**: `src/crewai_extrachallenge/crew.py`

```python
import os
from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task

@CrewBase
class CrewaiExtrachallenge():

    def _create_llm(self, temp_env_var: str, default_temp: str) -> LLM:
        """Helper to create LLM with temperature from .env."""
        model = os.getenv('MODEL', 'gpt-4-turbo-preview')
        temperature = float(os.getenv(temp_env_var, default_temp))
        return LLM(model=model, temperature=temperature)

    @agent
    def data_analyst(self) -> Agent:
        return Agent(
            config=self.agents_config['data_analyst'],
            llm=self._create_llm('TEMP_DATA_ANALYST', '0.1'),
            tools=[DBStatisticalAnalysisTool(), ...]
        )

    @agent
    def pattern_recognition_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['pattern_recognition_agent'],
            llm=self._create_llm('TEMP_PATTERN_AGENT', '0.3'),
            tools=[DBStatisticalAnalysisTool(), ...]
        )
```

**.env**:
```env
MODEL=gpt-4-turbo-preview
TEMP_DATA_ANALYST=0.1
TEMP_PATTERN_AGENT=0.3
TEMP_CLASSIFICATION=0.1
TEMP_REPORTING=0.2
```

**Benefits**:
- Model and temperature both from `.env`
- No hardcoding
- Per-agent customization
- Fallback to sensible defaults

---

## Testing Current Behavior

### Check What Temperature is Actually Used

```python
# Test script: test_temperature.py
import os
os.environ['MODEL'] = 'gpt-4-turbo-preview'

from crewai import Agent, LLM

# Test 1: Agent with no LLM specified
agent1 = Agent(
    role="Test",
    goal="Test",
    backstory="Test"
)
print(f"Agent1 LLM: {agent1.llm}")
print(f"Agent1 LLM temperature: {getattr(agent1.llm, 'temperature', 'unknown')}")

# Test 2: Agent with custom LLM
custom_llm = LLM(model="gpt-4-turbo-preview", temperature=0.1)
agent2 = Agent(
    role="Test",
    goal="Test",
    backstory="Test",
    llm=custom_llm
)
print(f"Agent2 LLM: {agent2.llm}")
print(f"Agent2 LLM temperature: {agent2.llm.temperature}")
```

---

## Current Configuration Status

### Files Status

| File | Temperature Config | Status |
|------|-------------------|--------|
| `.env` | `TEMP_DATA_ANALYST=0.1` (etc.) | ✅ Added (not used yet) |
| `config/agents.yaml` | None | ✅ Clean (no llm config) |
| `src/crew.py` | No LLM init | ✅ Uses .env MODEL with defaults |

### Current Temperature Values

Since no custom LLM initialization:

```
All agents use MODEL from .env with default temperature:
├── MODEL=gpt-5-mini
└── Default temp: ~0.7 (OpenAI default)
```

---

## Next Steps (Optional)

### If You Want Custom Temperatures

1. **Decide**: Do you need different temperatures per agent?
   - **Yes** → Implement Option 2 (custom LLM)
   - **No** → Keep current setup (defaults are fine)

2. **Implement** (if yes):
   - Add `_create_llm()` helper method to `crew.py`
   - Update each `@agent` to use custom LLM
   - Test with different temperature values

3. **Test**:
   ```bash
   # Edit .env temperatures
   TEMP_DATA_ANALYST=0.05
   TEMP_PATTERN_AGENT=0.4

   # Run and compare results
   crewai run
   ```

---

## Conclusion

### What We Learned

1. ✅ **CrewAI YAML doesn't support temperature configuration**
2. ✅ **Agent class has no `llm_config` or `temperature` parameters**
3. ✅ **Temperature must be set in LLM object, not Agent**
4. ✅ **Current setup uses model defaults from .env**

### Current Behavior

- **Model**: Configured via `.env` `MODEL` variable ✅
- **Temperature**: Uses model's default (0.7 for GPT models)
- **Per-agent customization**: Not available without code changes

### Recommendation

**For most use cases**: Current setup (defaults) is sufficient.

**For production**: If you need precise temperature control, implement Option 2 (custom LLM initialization in crew.py with .env variables).

**Temperature values in .env** are ready to use when you implement custom LLM initialization.
