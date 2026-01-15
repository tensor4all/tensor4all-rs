# Vibe Coding Workflow Guidelines

**Tools:** Cursor Pro

---

## 1. Ideation via Markdown
Start by drafting your initial ideas in natural language.
* Create a new `.md` file (e.g., `ideas.md` or `spec.md`).
* Write down the feature or functionality concept simply, as if explaining it to a colleague.

## 2. Preliminary Planning
Generate a concrete action plan using a **lightweight AI model** (e.g., Cursor's `auto` mode).
* **Prompt:** "Based on the ideas above, create a step-by-step coding plan."
* **Requirement:** Ensure the plan explicitly includes steps for **creating and running tests** before marking a task as complete.

## 3. Dual-Layer Review
Before writing code, validate the plan through a two-step review process to prevent structural errors.
1.  **Human Review:** Briefly check the plan for alignment with your original vision.
2.  **High-End AI Review:** Switch to a high-reasoning model (e.g., **GPT-5.2 Codex**) to review the plan.
    * *Goal:* Refine logic, catch edge cases, and optimize architecture.

## 4. Primary Implementation
Once the plan is validated, instruct the AI to begin implementation.
* **Default Model:** Use the **lightweight model** (e.g., Cursor's `auto` mode).
* **Focus:** Maximize speed and efficiency for standard boilerplate and logic.

## 5. Adaptive Model Switching
Monitor the AI's performance and complexity during implementation.
* **Trigger:** If the lightweight model becomes sluggish, repetitive, or struggles with complex logic.
* **Action:** Switch immediately to a **high-end model** (e.g., **GPT-5.2 Codex**) to resolve the bottleneck and ensure code quality.