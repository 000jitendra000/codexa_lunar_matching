class ImagePair:
    """
    Represents a pair of images (source and reference) for matching.
    """
    
    def __init__(self, source_info: dict, reference_info: dict, label: str = "unknown"):
        """
        Initializes an ImagePair.
        
        Args:
            source_info (dict): The loaded image dictionary for the source (e.g. from ImageLoader).
            reference_info (dict): The loaded image dictionary for the reference.
            label (str): The ground-truth relationship (e.g., 'positive', 'negative').
        """
        self.source = source_info
        self.reference = reference_info
        self.label = label

    @property
    def source_image(self):
        return self.source.get("image")
        
    @property
    def reference_image(self):
        return self.reference.get("image")
        
    def summary(self):
        """Returns a string summary of the pair."""
        src_shape = f"{self.source.get('width')}x{self.source.get('height')}x{self.source.get('channels')}"
        ref_shape = f"{self.reference.get('width')}x{self.reference.get('height')}x{self.reference.get('channels')}"
        return f"ImagePair [Label: {self.label}] | Source: {src_shape} -> Reference: {ref_shape}"
