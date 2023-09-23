
def get_sentences_from_document(document, model_func=None):
    """
    Get sentences from a document.

    :param document: document that need to split sentences
    :param model_func: function of sentence model, if specified, the
        function will be used for spliting document into different
        sentences.
    :return: document with the sentences separated by '\\\\n'
    """
    if model_func:
        sentences = model_func(document)
    else:
        sentences = document.splitlines()
    return '\n'.join(sentences)
