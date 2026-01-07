from langchain_community.document_loaders import JSONLoader

def load_ocr_json(path: str):
    jq_schema = """
    .parsing_res_list[]
    | select(.block_label == "text" or .block_label == "table")
    """

    loader = JSONLoader(
        file_path=path,
        jq_schema=jq_schema,
        text_content=False,
    )

    return loader.load()
