# XLens

A Library for Mechanistic Interpretability of Generative Language Models using JAX. Inspired by [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens).

## Overview

XLens is designed for mechanistic interpretability of Transformer language models, leveraging the power and efficiency of [JAX](https://github.com/jax-ml/jax). The primary goal of mechanistic interpretability is to reverse engineer the algorithms that a model has learned during training, enabling researchers and practitioners to understand the inner workings of generative language models.

## Features

⚠️ **Please Note:** Some features are currently in development and may not yet be fully functional. We appreciate your understanding as we work to improve and stabilize the library.

- **Support for Hooked Modules:** Interact with and modify internal model components seamlessly.
- **Model Alignment with Hugging Face:** Outputs from XLens are consistent with Hugging Face's implementation, making it easier to integrate and compare results.
- **Caching Mechanism:** Cache any internal activation for further analysis or manipulation during model inference. (Work in progress)
- **Full Type Annotations:** Comprehensive type annotations with generics and [jaxtyping](https://github.com/patrick-kidger/jaxtyping) for better code completion and type checking.
- **Intuitive API:** Designed with ease of use in mind, facilitating quick experimentation and exploration.

## Examples

TODO: Add examples
