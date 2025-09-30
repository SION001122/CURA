# CURA-based Model Example: `CURAv1_MODEL_NAME`

>  **This is an example template for CURA-based models intended for sharing on Hugging Face.**  
>  **Replace `CURAv1_MODEL_NAME` with your actual model name.**  
>  **Do not upload this file as-is without updating it for your own model.**

This example demonstrates how to publish a model based on the CURA architecture developed by **SION (2025–)**.

---

##  Model License

This model is subject to the **CURA NonCommercial Academic License v1.1**.

-  **Permitted**: Research, education, non-commercial sharing
-  **Prohibited**: Commercial use, resale, or integration into any paid product or service

If you redistribute or host a derivative model (e.g., on Hugging Face), you must:

- Set `license: other` or `license: custom`
- Include the following metadata:
  - `commercial_use: False`
  - `fine_tune_allowed: True`
  - `redistribute: True`
- Provide a visible link back to the original CURA repository:  
   [https://github.com/SION001122/CURA](https://github.com/SION001122/CURA)

---
Citation
If you use this model, please cite:
Jae-Bum Seo, Muhammad Salman, Lismer Andres Caceres-Najarro,  
“CURA: Size Isn’t All You Need – A Compact Universal Architecture for On-Device Intelligence”,  
arXiv:2509.24601, 2025. https://arxiv.org/abs/2509.24601


from transformers import AutoModel
model = AutoModel.from_pretrained("SION001122/CURAv1_MODEL_NAME")

 For commercial licensing, please contact: **sion@curalicense.org**
