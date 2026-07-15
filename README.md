# 🔎 SearcherKit

> Build, evaluate, and train search agents without locking your research to a
> single model, corpus, tool protocol, or training framework.

SearcherKit is a modular runtime for search-intensive agents. It brings agent
rollouts, heterogeneous search sources, tool execution, model adapters,
benchmark recipes, evaluation, and training integration into one coherent
stack.

<!-- ![figure2](figure2.png) -->

## ✨ Highlights

- 🧠 **Search agent development, all-in-one** — Covers scaffold, training, evaluation & user interface. Kickstart your search agent development

![SearcherKit](docs/images/searcherkit.png){width=60% fig-align="center"}

- 🌐 **Retrieve from various sources** — Integrate with search sources from files, knowledge bases and web pages. Build and use your local knowledge bases with a few lines of command.

![SearcherKit supports multiple search sources](docs/images/sources.png){width=60% fig-align="center"}

- ⚡ **Lightweight, efficient and developer friendly** — `SearcherKit` focus on scaffold/harness level without heavy dependencies. Can be easily integrated into your existing projects as a library. Invoke `searcherkit` functions with full async execution pipeline to accelerate your project.

## 🚀 Quick Start

Install the project in editable mode:

```bash
pip install -e .
```

The default installation includes the Elasticsearch source, Anthropic client,
and terminal UI dependencies. The `indexing` extra remains available for
building and deploying local indexes:

```bash
pip install -e '.[indexing]'
```

See [Quick Start](docs/index.qmd) to start running your search agent.

## 📖 Documentation

- [Full guide](docs/index.qmd)
- [CLI reference](docs/cli/index.qmd)

## 🤝 Contributing

SearchAgent is actively evolving. Reproduction reports, new model parsers,
source adapters, benchmark recipes, training integrations, bug reports, and
documentation improvements are welcome. Open an issue with the model,
benchmark, and environment you want to support—or submit a focused pull
request.

If SearchAgent helps your research or makes your agent stack easier to reason
about, consider starring the repository. It helps more search-agent builders
find the project.

## 📜 License

**TODO**: Add license information.
