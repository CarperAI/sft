# Orca

## Generation

First, set your openai API key as an environment variable:

```bash
export OPENAI_API_KEY=...
```

Then, run the generation script:

```bash
python generate_dataset.py
```

Note: this will take a while, and will use up quite a bit of cash. It's recommended to set a smaller amount for testing.

You can do this by editing line 22-29 of `generate_dataset.py`:

```python
cot_total = 15
cot_gpt4_total = cot_total//5
niv_total = 44
niv_gpt4_total = niv_total//5
flan_total = 250
flan_gpt4_total = flan_total//5
t0_total = 200
t0_gpt4_total = t0_total//5
```
for a quick example.