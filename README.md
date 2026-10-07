![AI agent context window management: summarize, trim, and delete messages](assets/agent-context-window-banner.png)

# Stop Context Overload!

### AI Agent Memory Management with LangChain & LangGraph

Learn how to manage an AI agent’s conversation history through message summarization, trimming, and deletion. This lecture walks through Python examples using LangChain middleware and LangGraph short-term memory.

## Three approaches

| Approach | What it does | Example in the lecture |
| --- | --- | --- |
| **Summarize** | Compress older messages into a concise summary to preserve key context. | `SummarizationMiddleware`, extended with logging to show when summarization happens. |
| **Trim** | Keep selected messages and remove the rest to fit the context window. | A `before_model` hook that keeps the first message and the three most recent messages. |
| **Delete** | Remove specific messages from the agent’s stored conversation history. | An `after_model` hook that removes messages at indices 1 and 2 when the history contains more than five messages. |

## Code walkthrough

- Create a tool-using agent with weather and internet search tools.
- Store conversation history with `InMemorySaver` and a thread ID.
- Configure summarization with a message-count trigger and a token-based keep setting.
- Replace message history using `RemoveMessage` and `REMOVE_ALL_MESSAGES` for trimming.
- Delete individual messages by their IDs.
- Start a fresh conversation with `/new`, or leave the chat with `bye` or `exit`.

The provided example enables **message deletion**. The summarization and trimming middleware are included as commented alternatives for demonstration. Its summarization configuration triggers at four messages and uses a 300-token keep setting.

## Tools and libraries

- **Python** for the implementation.
- **LangChain** for agent creation, tools, and middleware.
- **LangGraph** for in-memory checkpointing and conversation state.
- **Ollama, Groq, and OpenAI** for model access and fallback examples.
- **OpenWeatherMap, Tavily, and DuckDuckGo** for tool demonstrations.
- **python-dotenv** for loading local environment variables.

## Local configuration

The example reads `GROQ_API_KEY`, `OPENWEATHER_API_KEY`, and `TAVILY_API_KEY` from the environment. OpenAI model access also requires `OPENAI_API_KEY`; the primary Ollama model must be available locally.

`InMemorySaver` keeps conversation history within the running process. Each `/new` command creates a new thread ID and starts a separate conversation.
