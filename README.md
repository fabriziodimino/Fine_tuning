🚀 𝐁𝐮𝐢𝐥𝐝𝐢𝐧𝐠 𝐚𝐧 𝐀𝐢𝐫𝐛𝐧𝐛 𝐂𝐡𝐚𝐭𝐛𝐨𝐭 𝐰𝐢𝐭𝐡 𝐚 𝐅𝐢𝐧𝐞-𝐓𝐮𝐧𝐞𝐝 𝐋𝐋𝐌 🚀



Last week, a friend reached out overwhelmed by Airbnb requests. My solution? A specialized chatbot fine-tuned on Meta LLama 3.2 (3B parameters) to streamline inquiries and improve guest communication.



I’ve open-sourced the entire codebase on GitHub—download the fine-tuned model, run it locally with Ollama, and see how effectively a custom LLM handles Airbnb-related questions.



Here’s how I did it:



1. 𝐃𝐚𝐭𝐚𝐬𝐞𝐭 𝐂𝐫𝐞𝐚𝐭𝐢𝐨𝐧

- Extracted key data from a property PDF using IBM Deep Search (https://github.com/DS4SD/docling).

- Created precise Q&A pairs for robust training with OpenAI 4o-mini.



2. 𝐅𝐢𝐧𝐞-𝐭𝐮𝐧𝐢𝐧𝐠

- Leveraged the Unsloth AI library for up to 2x faster fine-tuning on Llama 3.2 (https://github.com/unslothai/unsloth).

- Employed 𝐋𝐨𝐑𝐀 (Low-Rank Adaptation) to freeze most weights and learn only essential parameters.

- Utilized the Hugging Face SFFT Trainer for an efficient, organized training pipeline (https://github.com/huggingface/trl).
