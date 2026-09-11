# Shared Preprocessing Rules

Both members must use the same rules:
1. Combine `subject + message`.
2. Lowercase.
3. Replace URLs with `URLTOKEN`.
4. Replace email addresses with `EMAILTOKEN`.
5. Replace numbers with `NUMTOKEN`.
6. Normalize whitespace.
7. Use the same train/test split configuration from `config/settings.py`.
