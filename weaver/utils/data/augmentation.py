import numpy as np
import awkward as ak
import copy
import logging

_logger = logging.getLogger('weaver')

class Augmenter(object):
    _logged_init = False  # Class variable to log only once
    _fetch_count = 0  # Count fetches for logging
    
    def __init__(self, data_config, options=None):
        self.data_config = data_config
        self.options = options if options is not None else {}
        
        # Default augmentation settings (can be overridden by options)
        self.enabled = self.options.get('augment', False)
        self.params = {
            'rotation': self.options.get('aug_rotation', True),
            'reflection': self.options.get('aug_reflection', True),
            'dropout': self.options.get('aug_dropout', 0.0),  # Fraction of particles to drop (0.0-1.0)
            'reflection_prob': self.options.get('aug_reflection_prob', 0.5),  # Probability to apply each reflection
        }
        
        # Store last rotation angle and reflection stats for logging
        self._last_rotation_angle = None
        self._last_phi_flip_pct = None
        self._last_eta_flip_pct = None
        
        # Log augmentation settings (only once per process)
        if not Augmenter._logged_init and self.enabled:
            _logger.info('[Augmenter] Initialized with: rotation=%s, reflection=%s (prob=%.2f), dropout=%.2f',
                        self.params['rotation'], self.params['reflection'], 
                        self.params['reflection_prob'], self.params['dropout'])
            Augmenter._logged_init = True

    def augment(self, table):
        if not self.enabled:
            return table
        
        Augmenter._fetch_count += 1
        n_events = len(table)
        
        if self.params['rotation']:
            table = self._rotate(table)
            
        if self.params['reflection']:
            table = self._reflect(table)
            
        if self.params['dropout'] > 0:
            table = self._dropout(table)
        
        # Log EVERY fetch with augmentation summary
        rot_info = f"rot={self._last_rotation_angle:.3f}rad" if self._last_rotation_angle is not None else "rot=N/A"
        phi_info = f"phi_flip={self._last_phi_flip_pct:.1f}%" if self._last_phi_flip_pct is not None else ""
        eta_info = f"eta_flip={self._last_eta_flip_pct:.1f}%" if self._last_eta_flip_pct is not None else ""
        
        _logger.info(f'[Augmenter] Fetch #{Augmenter._fetch_count}: {n_events} events | {rot_info} | {phi_info} | {eta_info}')
            
        return table

    def _rotate(self, table):
        """
        Azimuthal rotation around beam axis.
        Rotates all particles by random angle theta in [0, 2π).
        This is a physics-preserving transformation.
        """
        # Check if we have the necessary columns
        if 'pfcand_phirel' not in table.fields:
            self._last_rotation_angle = None
            return table
            
        # Generate random angle for each event
        n_events = len(table['pfcand_phirel'])
        theta = np.random.uniform(0, 2 * np.pi, n_events)
        
        # Store first rotation angle for logging
        self._last_rotation_angle = float(theta[0]) if n_events > 0 else None
        
        # Broadcast theta to match particle structure
        # theta is (N,), table['pfcand_phirel'] is (N, M) where M can be jagged
        theta_broadcast = ak.broadcast_arrays(theta, table['pfcand_phirel'])[0]
        
        # Update phirel
        new_phirel = table['pfcand_phirel'] + theta_broadcast
        
        # Wrap to [-pi, pi]
        new_phirel = (new_phirel + np.pi) % (2 * np.pi) - np.pi
        table['pfcand_phirel'] = new_phirel
        
        # Rotate px, py if they exist (cartesian coordinates)
        if 'pfcand_px' in table.fields and 'pfcand_py' in table.fields:
            px = table['pfcand_px']
            py = table['pfcand_py']
            
            c = np.cos(theta_broadcast)
            s = np.sin(theta_broadcast)
            
            new_px = px * c - py * s
            new_py = px * s + py * c
            
            table['pfcand_px'] = new_px
            table['pfcand_py'] = new_py
        
        # Update drrel since phirel changed
        # drrel = sqrt(thetarel^2 + phirel^2)
        if 'pfcand_drrel' in table.fields and 'pfcand_thetarel' in table.fields:
            table['pfcand_drrel'] = np.sqrt(
                table['pfcand_thetarel']**2 + table['pfcand_phirel']**2
            )
            
        return table

    def _reflect(self, table):
        """
        Reflections across symmetry planes.
        - Phi reflection (φ → -φ): Reflects across φ = 0 plane
        - Eta reflection (θ → -θ): Reflects across η = 0 plane
        
        Each reflection is applied independently with probability reflection_prob.
        This is a physics-preserving transformation for symmetric detectors.
        """
        n_events = len(table)
        reflection_prob = self.params['reflection_prob']
        
        # 1. Phi reflection: phirel -> -phirel
        # Apply to each event with probability reflection_prob
        apply_phi_reflection = np.random.rand(n_events) < reflection_prob
        mask_phi = np.where(apply_phi_reflection, -1, 1)
        
        # Store phi flip stats for logging
        self._last_phi_flip_pct = 100.0 * np.sum(apply_phi_reflection) / n_events if n_events > 0 else 0
        
        if 'pfcand_phirel' in table.fields:
            mask_phi_broadcast = ak.broadcast_arrays(mask_phi, table['pfcand_phirel'])[0]
            table['pfcand_phirel'] = table['pfcand_phirel'] * mask_phi_broadcast
            
            # Update py (flips sign with phi reflection)
            if 'pfcand_py' in table.fields:
                table['pfcand_py'] = table['pfcand_py'] * mask_phi_broadcast

        # 2. Eta reflection: thetarel -> -thetarel
        # Apply to each event with probability reflection_prob (independent of phi reflection)
        apply_eta_reflection = np.random.rand(n_events) < reflection_prob
        mask_eta = np.where(apply_eta_reflection, -1, 1)
        
        # Store eta flip stats for logging
        self._last_eta_flip_pct = 100.0 * np.sum(apply_eta_reflection) / n_events if n_events > 0 else 0
        
        if 'pfcand_thetarel' in table.fields:
            mask_eta_broadcast = ak.broadcast_arrays(mask_eta, table['pfcand_thetarel'])[0]
            table['pfcand_thetarel'] = table['pfcand_thetarel'] * mask_eta_broadcast
            
            # Update pz (flips sign with eta reflection)
            if 'pfcand_pz' in table.fields:
                table['pfcand_pz'] = table['pfcand_pz'] * mask_eta_broadcast
                
            # Update dz (longitudinal impact parameter)
            if 'pfcand_dz' in table.fields:
                table['pfcand_dz'] = table['pfcand_dz'] * mask_eta_broadcast
        
        # Update drrel since phirel and/or thetarel may have changed
        # Note: drrel = sqrt(thetarel^2 + phirel^2), so sign flips don't affect it,
        # but we recalculate for consistency
        if 'pfcand_drrel' in table.fields and 'pfcand_thetarel' in table.fields and 'pfcand_phirel' in table.fields:
            table['pfcand_drrel'] = np.sqrt(
                table['pfcand_thetarel']**2 + table['pfcand_phirel']**2
            )

        return table

    def _dropout(self, table):
        """
        Soft Particle Dropout: Preferentially removes lowest-pT particles.
        
        This simulates detector inefficiency and provides regularization.
        Drops approximately 'dropout' fraction of particles, biased toward low pT.
        
        Strategy: For each particle, compute dropout probability inversely proportional to pT.
        This naturally targets soft particles while occasionally dropping harder ones.
        """
        if 'pfcand_mask' not in table.fields:
            return table
            
        frac_to_drop = self.params['dropout']
        if frac_to_drop <= 0:
            return table
        
        # Need pT information to bias dropout toward low-pT particles
        if 'pfcand_pt_log' not in table.fields:
            return table
            
        pt_log = table['pfcand_pt_log']
        mask = table['pfcand_mask']
        
        # Strategy: Dropout probability ~ 1/pT
        # We use pt_log, so smaller values = lower pT
        # Convert to linear scale for dropout probability
        # prob_drop = base_prob * exp(-pt_log) / normalization
        
        # Simpler approach: Use softmax-like weighting
        # For each event, we want to drop ~frac_to_drop of particles, biased to low pT
        
        # Method: Dropout probability = sigmoid(-α * (pt_log - median_pt_log))
        # This makes particles below median more likely to drop
        
        # Even simpler: Drop with probability proportional to ranking
        # Particles in bottom X percentile have higher dropout probability
        
        # SIMPLEST & EFFECTIVE: 
        # Create dropout probability that's higher for low pT:
        # prob = base_prob * (max_pt - pt) / (max_pt - min_pt)
        
        # Convert pt_log to dropout weight (higher weight = more likely to drop)
        # Use inverted normalized pT: lower pT -> higher weight
        
        # For each event separately, create weights
        # This is tricky with awkward arrays, so we use a simpler approach:
        
        # Create per-particle dropout probability inversely proportional to pT
        # First normalize pt_log per event (to handle different jet pT spectra)
        
        # Compute per-event statistics
        valid_mask = mask > 0
        
        # For numerical stability and to bias toward low-pT:
        # dropout_prob_i = frac_to_drop * 2 * (1 - (pt_i - pt_min) / (pt_max - pt_min))
        # This gives prob=2*frac for lowest pT, prob=0 for highest pT, average=frac
        
        # Compute min/max pT per event (only for valid particles)
        # Replace masked values with large number so they don't affect min/max
        pt_log_masked = ak.where(valid_mask, pt_log, 999)
        pt_min = ak.min(pt_log_masked, axis=1, keepdims=True)
        
        pt_log_masked_for_max = ak.where(valid_mask, pt_log, -999)
        pt_max = ak.max(pt_log_masked_for_max, axis=1, keepdims=True)
        
        # Normalize to [0, 1] where 0 = lowest pT, 1 = highest pT
        pt_range = pt_max - pt_min
        pt_range = ak.where(pt_range > 0, pt_range, 1.0)  # Avoid division by zero
        
        pt_normalized = (pt_log - pt_min) / pt_range
        
        # Invert: low pT -> high value
        pt_weight = 1.0 - pt_normalized
        
        # Create dropout probability: scale so mean ~ frac_to_drop
        # prob = 2 * frac_to_drop * pt_weight
        # This gives prob in [0, 2*frac], with mean = frac
        dropout_prob = 2.0 * frac_to_drop * pt_weight
        
        # Clip to [0, 1]
        dropout_prob = ak.where(dropout_prob > 1.0, 1.0, dropout_prob)
        dropout_prob = ak.where(dropout_prob < 0.0, 0.0, dropout_prob)
        
        # Generate random numbers
        rand = ak.unflatten(np.random.rand(ak.count(mask)), ak.num(mask))
        
        # Drop where rand < dropout_prob
        drop_mask = rand < dropout_prob
        
        # Update mask: set to 0 if dropped (but only for already-valid particles)
        new_mask = ak.where(drop_mask & valid_mask, 0, mask)
        table['pfcand_mask'] = new_mask
        
        # Zero out features for dropped particles for safety
        # (Model should respect mask, but this ensures consistency)
        for col in table.fields:
            if col.startswith('pfcand_') and col != 'pfcand_mask':
                try:
                    # Only zero out if it's a numerical array with same shape as mask
                    if hasattr(table[col], 'ndim'):
                        table[col] = table[col] * new_mask
                except:
                    pass  # Skip if shapes don't match or other issues
                    
        return table
