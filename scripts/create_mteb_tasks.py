"""
This script creates MTEB classes based off of our indexed task primitive.

It's not clear if we should continue to extend MTEB tasks this way. While this repo matures,
this is beneficial for explicit separation of concerns. If you use our tasks, you know you're
using our abstractions.

In the long term, it doesn't seem sustainable to depend on MTEB tasks this way. Eventually, we
should have a clear idea of what could be upstreamed into MTEB.

REPS should grow to also benchmark metrics important for production, e.g. latency and cost.
I want to find a way to upstream some of these changes.
"""
from mteb import AbsTaskRetrieval

CLASS_PREFIX = "MTEB"

def main():
    print("creating mteb tasks.")
    path = "reps/evaluation/mteb_tasks.py"

    output = []
    names = []
    output.append("from reps.evaluation.indexed_task import IndexedTask")
    output.append("")
    for cls in AbsTaskRetrieval.__subclasses__():
        names.append(cls.__name__)
        code = [
            f"class {cls.__name__}(IndexedTask,{CLASS_PREFIX}{cls.__name__}):",
            "    def __init__(self, **kwargs):",
            "        super().__init__(**kwargs)",
            "",
        ]
        output += code

    # import all of the tasks.
    import_statement = ["from mteb import ("]
    for name in names:
        import_statement.append(f"    {name} as {CLASS_PREFIX}{name},")

    import_statement.append(")")

    output = import_statement + output
    # add line breaks to all output.
    output = [x+"\n" for x in output]

    with open(path, "w+") as f:
        f.writelines(output)

if __name__ == "__main__":
    main()
