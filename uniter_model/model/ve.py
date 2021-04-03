"""
UNITER for VE model
"""
from .vqa import UniterForVisualQuestionAnswering


class UniterForVisualEntailment(UniterForVisualQuestionAnswering):
    """ Finetune multi-modal BERT for VE
    """
    def __init__(self, config, img_dim):
        super().__init__(config, img_dim, 3)
