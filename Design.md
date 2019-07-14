# Design of SPMDfy

## Pipeline
1. Get the filename 
2. Parse with libTooling
3. Get the handle to Translation Unit(TU)
4. Parse all the kernel declarations
    1. For each declaration
        1. Construct a Control Flow Graph(CFG)
            1. Form graph from statment(what is a statement???)