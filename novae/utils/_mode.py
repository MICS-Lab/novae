class Mode:
    """Novae mode class, used to store states variables related to training and inference."""

    all_clustering_attrs: list[str] = ["_clustering", "_clustering_zero", "_clusters_levels", "_clusters_levels_zero"]

    def __init__(self):
        self.use_queue = False
        self.queue_mode = False
        self.zero_shot = False
        self.freeze_mode = True
        self.trained = False

    def __repr__(self) -> str:
        return f"Mode({dict(self.__dict__.items())})"

    ### Mode modifiers

    def pretrained(self):
        self.use_queue = False
        self.freeze_mode = False
        self.queue_mode = False
        self.trained = True
        self.zero_shot = False

    def fine_tune(self):
        self.use_queue = False
        self.queue_mode = False
        self.freeze_mode = False
        self.zero_shot = False

    def fit(self):
        self.zero_shot = False
        self.trained = False

    # Mode-specific attributes

    @property
    def clustering_attr(self):
        return "_clustering_zero" if self.zero_shot else "_clustering"

    @property
    def clusters_levels_attr(self):
        return "_clusters_levels_zero" if self.zero_shot else "_clusters_levels"
