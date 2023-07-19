# TextObfuscator
Code for Findings of ACL 2023 **"[TextObfuscator: Making Pre-trained Language Model a Privacy Protector via Obfuscating Word Representations
](https://aclanthology.org/2023.findings-acl.337)"**

Please kindly cite the paper if you use the code or any resources in this repo:
```bib
@inproceedings{zhou-etal-2023-textobfuscator,
    title = "{T}ext{O}bfuscator: Making Pre-trained Language Model a Privacy Protector via Obfuscating Word Representations",
    author = "Zhou, Xin  and
      Lu, Yi  and
      Ma, Ruotian  and
      Gui, Tao  and
      Wang, Yuran  and
      Ding, Yong  and
      Zhang, Yibo  and
      Zhang, Qi  and
      Huang, Xuanjing",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.337",
    pages = "5459--5473",
    abstract = "In real-world applications, pre-trained language models are typically deployed on the cloud, allowing clients to upload data and perform compute-intensive inference remotely. To avoid sharing sensitive data directly with service providers, clients can upload numerical representations rather than plain text to the cloud. However, recent text reconstruction techniques have demonstrated that it is possible to transform representations into original words, suggesting that privacy risk remains. In this paper, we propose TextObfuscator, a novel framework for protecting inference privacy by applying random perturbations to clustered representations. The random perturbations make the representations indistinguishable from surrounding clustered representations, thus obscuring word information while retaining the original word functionality. To achieve this, we utilize prototypes to learn clustered representation, where tokens of similar functionality are encouraged to be closer to the same prototype during training.Additionally, we design different methods to find prototypes for token-level and sentence-level tasks, which can improve performance by incorporating semantic and task information.Experimental results on token and sentence classification tasks show that TextObfuscator achieves improvement over compared methods without increasing inference cost.",
}
```
