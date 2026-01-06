from paddleocr import PaddleOCRVL
import os

file_path = r"D:/github/DocMind-RAG/data/PrintingTechniqueDataset/image/01/print-tech-outlier-01-002.png"

if not os.path.exists(file_path):
    print("file is not exist")
    exit()

pipeline = PaddleOCRVL()

# output = pipeline.predict(file_path,
#                           use_doc_preprocessor=True,
#                           use_layout_detection=True,
#                           merge_layout_blocks=False,
#                           format_block_content=True,
#                           markdown_ignore_labels=[])

output = pipeline.predict(file_path)

for res in output:
    res.print()
    res.save_to_json(save_path="../../data/PrintingTechniqueDataset_json")    
    # res.save_to_markdown(save_path="output")
