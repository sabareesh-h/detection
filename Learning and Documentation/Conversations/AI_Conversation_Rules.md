# AI Assistant Instructions: Exporting Chat Transcripts

Whenever the user asks to "save" or "export" the conversation, you (the AI) must follow these exact rules without the user needing to repeat them:

### 1. Where to Place the File
Always save the conversation transcript directly into the following folder:
`c:\Users\RohithSuryaCKM\Downloads\Projects\Image_detection\Learning and Documentation\Conversations\`

### 2. Naming the File
Name the file starting with `conversation_` followed by a short description of the chat topic. 
Example: `dataset_splitting.md` or `yolo_training.md`.

### 3. Formatting
The file should be a clean, word-for-word transcript that matches the chat interface. Do not summarize unless asked. 
Format it like this:

---
**👤 User** 
[User's exact message]

---
**🤖 Antigravity** 
[AI's exact response]

---

### 4. Execution
Simply use the `write_to_file` tool to create the markdown file using the rules above, write the full history into it, and notify the user when done!
