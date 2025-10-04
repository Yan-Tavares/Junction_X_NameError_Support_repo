import numpy as np
from typing import List, Tuple


class ModelEnsemble:
    """Ensemble multiple models and combine their predictions."""
    
    def __init__(self, models):
        """Initialize the ensemble with a list of models.
        
        Args:
            models: List of model instances that have a predict method
        """
        self.models = models
    
    def predict(self, audio_sample):
        """Ensemble predictions from all models.
        
        Args:
            audio_sample: Audio data to process
            
        Returns:
            List of timestamps after ensembling
        """
        all_predictions = []
        
        # Collect predictions from all models
        for model in self.models:
            timestamps = model.predict(audio_sample)
            all_predictions.extend(timestamps)
        
        # Combine predictions by clustering nearby timestamps
        # and taking the average of clusters
        if not all_predictions:
            return []
        
        # Sort all predictions
        all_predictions = sorted(all_predictions)
        
        # Simple clustering: group timestamps within 2 seconds of each other
        ensembled_timestamps = []
        current_cluster = [all_predictions[0]]
        
        for ts in all_predictions[1:]:
            if ts - current_cluster[-1] <= 2.0:  # Within 2 seconds
                current_cluster.append(ts)
            else:
                # Average the current cluster and start a new one
                ensembled_timestamps.append(np.mean(current_cluster))
                current_cluster = [ts]
        
        # Don't forget the last cluster
        if current_cluster:
            ensembled_timestamps.append(np.mean(current_cluster))
        
        return sorted(ensembled_timestamps)
    
    def predict_with_voting(self, audio_sample, threshold=1.5):
        """Ensemble with voting: only keep timestamps that multiple models agree on.
        
        Args:
            audio_sample: Audio data to process
            threshold: Distance threshold for considering timestamps as "votes" for same event
            
        Returns:
            List of timestamps that have multiple model votes
        """
        all_predictions = []
        
        # Collect predictions from all models
        for model in self.models:
            timestamps = model.predict(audio_sample)
            all_predictions.append(timestamps)
        
        if not all_predictions:
            return []
        
        # Flatten and sort
        flat_predictions = sorted([ts for pred_list in all_predictions for ts in pred_list])
        
        if not flat_predictions:
            return []
        
        # Count votes for each timestamp
        voted_timestamps = []
        
        for ts in flat_predictions:
            # Count how many models have a prediction near this timestamp
            votes = sum(
                1 for pred_list in all_predictions
                if any(abs(t - ts) <= threshold for t in pred_list)
            )
            
            # Only keep if at least 2 models agree
            if votes >= 2:
                voted_timestamps.append(ts)
        
        # Remove duplicates by clustering again
        if not voted_timestamps:
            return []
        
        final_timestamps = []
        current_cluster = [voted_timestamps[0]]
        
        for ts in voted_timestamps[1:]:
            if ts - current_cluster[-1] <= threshold:
                current_cluster.append(ts)
            else:
                final_timestamps.append(np.mean(current_cluster))
                current_cluster = [ts]
        
        if current_cluster:
            final_timestamps.append(np.mean(current_cluster))
        
        return sorted(final_timestamps)
