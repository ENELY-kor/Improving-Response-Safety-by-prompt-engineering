# Improving-Response-Safety-by-prompt-engineering
LLM, Prompt Engineering, AI-Safety
# Overview
This project focuses on maximizing the safety of Large Language Model (LLM) responses using purely Prompt Engineering techniques, without relying on GPU computation. In an increasingly AI-dependent society, ensuring the stability and safety of generated output—especially concerning sensitive topics like discrimination—is a critical challenge.
* Core Objective: Enhance LLM response safety through prompt engineering to evade dangerous inquiries and provide alternative, safe responses.
* Evaluation Metric: Safety scoring measured by the OffensiveLanguageClassifier.
* Key Challenge Addressed: Resolving the issue of spuriously high safety scores caused by the model's unintended 'shortcuts' resulting in uniform, overly positive responses.
The primary goal is to create an LLM that not only avoids answering risky questions but also effectively bypasses hazardous elements and offers constructive alternatives. The level of enhanced safety is quantitatively assessed using an OffensiveLanguageClassifier.
# Methodology & Implementation
To secure safety and prevent the model from taking undesirable 'shortcuts,' we implemented a multi-stage strategy involving refined prompt engineering and sophisticated post-processing techniques.
* Initial Problem

Using strong, clear role-based prompts (Hard Prompts) initially led to the model generating responses that were too uniform and positive, resulting in artificially inflated safety scores ('shortcuts').
* Solution

 Prompt Softening: 
 
 The role assignment was kept as light as possible to increase the model's freedom and prevent rigid adherence to the initial safety instructions.

 Sampling Technique: 
 
 Top-p sampling was employed to generate a maximum diversity of Candidate Responses. This was crucial for moving beyond the uniform positive output and exposing potential vulnerabilities for realistic safety auditing.

* Enhancement dialog

To minimize Context Deviation and ensure the final output addresses the user's original intent, we selected the candidate response with the highest semantic similarity to the original query.

 Lightweight Embedding Model: 
 
 A compact model, such as all-MiniLM-L6-v2, was used to calculate the Semantic Similarity between each candidate response and the original user question.

 Response Selection: 
 
 The candidate with the highest similarity score was chosen as the final, relevant response.

 Evasive Query Blocking: 
 
 Even if a user's question is indirect, it is checked for harmful terms using the function contains_hate_speech(user_input). If harmful content is detected, the response generation is proactively blocked.

 Final Response Analysis: 
 
 The generated final response, reply_txt, is checked for the presence of harmful language using contains_hate_speech(reply_txt).

 Handling: 
 
 Detected unsafe responses are either rejected or safely sanitized using the sanitize_response(reply_txt) function before final output.
# Result
![Result](images/result.png)
# Key Insights & Reflection
* Insights on Prompt Engineering

**The Trap of Shortcuts:** We confirmed that overly rigid prompts can lead to a model taking 'shortcuts'—mechanically generating responses that appear safe but reflect an artificial, non-robust safety score. Techniques like Top-p sampling were essential to expose the model's true vulnerabilities and develop effective countermeasures.

**Defense-in-Depth:** Prompting alone is insufficient for complete safety. A multi-layered architecture combining pre-filtering of inputs and post-processing sanitation/rejection of outputs is crucial for robust safety in a real-world service environment.

* Future Direction

**Maximizing Zero-Shot Learning:** Future work will focus on combining prompt engineering with methods for storing and loading safety configurations from 'memory' (e.g., creating a guideline repository for efficient Context Window usage) to maximize the effectiveness of Zero-shot Learning.
