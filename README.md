# 🔎 SearcherKit

SearcherKit is a modular system for building AI agents that rely heavily on search. It handles everything from agent rollouts, LLM backends, tool calls, and benchmark evaluation, to training pipelines — all in one package.



![SearcherKit](docs/images/searcherkit_1.png)


## What it does

- **Build and run search agents** – Define agent behavior once, then use it for prototyping, evaluation, interactive demos, and large‑scale rollout generation.
- **Connect to any data source** – Plug in web APIs, Elasticsearch, internal knowledge bases, or local files through a unified interface. Combine sources freely without touching agent code.
- **Scale without pain** – Async execution, flexible concurrency, checkpoint recovery, and detailed traces make batch jobs fast and debuggable. Swap models and backends easily.
- **Training‑ready** – The same agent and tool configuration can be used for evaluation, offline analysis, and integration with training pipelines, without needing to refactor code.

---

## Why SearcherKit?

- 🧩 **One codebase, many uses** – No more copying logic between research and production. What you build for an experiment works directly for evaluation and deployment.
- 🔌 **Plug‑and‑play sources** – Add or remove search backends in minutes. The agent doesn’t care where the data comes from.
- ⚡ **Fast and reliable** – Native async, smart concurrency, and automatic checkpointing let you run weeks‑long experiments without babysitting.
- 🏆 **Proven in RL** – We’ve used it to train competitive 8B models on real search tasks. The same pipeline is open and ready for your own training.

---

### 🚀 Quick Start

Jump right in with the [Quick Start guide](docs/guide/index.qmd) — you’ll have a search agent running in minutes.

### 📖 Documentation

Browse the docs by topic:

- **Getting started** – [Quick Start](docs/guide/index.qmd) · [Searching files](docs/guide/01-searching/01-search-files.qmd) · [Searching webpages](docs/guide/01-searching/02-search-web.qmd)  
- **Training** – [Overview of SFT & RL workflows](docs/guide/02-training/01-training-overview.qmd)  
- **Customization** – [Project architecture and extending components](docs/guide/03-customizing/01-project-architecture.qmd)  
- **Interfaces** – [CLI reference](docs/guide/04-interface/01-use-the-cli-interface.qmd) · [Interactive TUI](docs/guide/04-interface/02-use-the-interactive-tui.qmd)


---

### 🤝 Contributing

SearcherKit is under active development. We welcome issues and pull requests for reproduction reports, new model parsers, source adapters, benchmark recipes, training integrations, bug fixes, documentation, and more.

If SearcherKit helps your research or simplifies your agent stack, please give us a ⭐ on GitHub – it helps others discover the project.

### 👤 The Team

SearcherKit is built and maintained by the group of Assistant Professor [Wei Hongxin](https://hongxin001.github.io/) and Professor [Jing Bingyi](https://hongxin001.github.io/) in SUSTech and CUHK-SZ.  
Core maintainers: [Li Hanyang](https://justinliii.github.io/), Zhang Haotian, Yang Hanjie, Wang Shuoyuan, Lan Zijie, and Yu Zhengye.

### 📜 License

SearcherKit is licensed under the [MIT License](LICENSE).
