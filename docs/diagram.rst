Example Mermaid Diagram
=======================

This page demonstrates a minimal, reproducible Mermaid diagram
rendered using **reStructuredText** in Sphinx / Read the Docs.

Workflow Diagram
----------------

The following diagram shows a simple data-processing workflow.

.. mermaid::

   flowchart TD
       A[Raw Input Data]
       B[Validation]
       C[Processing]
       D[Output Files]
       E[Error Handling]

       A --> B
       B -->|Valid| C
       B -->|Invalid| E
       C --> D