"""Experiment-specific report orchestration modules.

Available reports:
- analyze_pid_embeddings: Deep analysis of PID embedding representations, geometry, and evolution
- analyze_representations: Deep analysis of transformer representations
- compare_embeddings_4tbg: Compare embedding strategies in emb_pe_4tbg sweep
- compare_globals_heads: Compare globals reconstruction weights
- compare_model_sizes: Compare model size and architecture effects
- compare_norm_pos_pool: Compare normalization, positional encoding, and pooling
- compare_overfitting_regularization_sweep: Overfitting/regularization analysis
- compare_positional_encodings: Compare positional encoding strategies
- compare_regularization: Compare regularization hyperparameters
- compare_tokenizers: Compare VQ vs non-VQ tokenizers
- evaluate_classifier: Basic classifier evaluation with inference
- phd_summary_4tbg: High-level summary of emb_pe_4tbg sweep (embedding, PE, size)
- phd_summary_binning_vs_direct: Summary of binning vs direct experiment (pooling, tokenization, MET, vect)
"""
