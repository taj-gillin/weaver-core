import unittest
import numpy as np
import awkward as ak
import sys
import os

# Add parent directory to path to import weaver modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from weaver.utils.data.augmentation import Augmenter

class TestAugmentation(unittest.TestCase):
    def setUp(self):
        # Create mock data
        # 2 events, 3 particles each
        self.n_events = 2
        self.n_particles = 3
        
        self.phirel = ak.Array([
            [0.1, 0.2, 0.3],
            [-0.1, -0.2, -0.3]
        ])
        
        self.thetarel = ak.Array([
            [0.1, 0.2, 0.3],
            [0.1, 0.2, 0.3]
        ])
        
        self.px = ak.Array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])
        
        self.py = ak.Array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0]
        ])
        
        self.mask = ak.Array([
            [1, 1, 1],
            [1, 1, 1]
        ])
        
        # Compute drrel
        self.drrel = np.sqrt(self.thetarel**2 + self.phirel**2)
        
        self.table = ak.Array({
            'pfcand_phirel': self.phirel,
            'pfcand_thetarel': self.thetarel,
            'pfcand_px': self.px,
            'pfcand_py': self.py,
            'pfcand_pz': ak.Array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            'pfcand_mask': self.mask,
            'pfcand_pt_log': ak.Array([[1.0, 0.5, 0.1], [0.8, 0.3, 0.05]]),  # Varying pT for dropout test
            'pfcand_drrel': self.drrel
        })
        
        self.data_config = {} # Mock config

    def test_rotation(self):
        options = {
            'augment': True,
            'aug_rotation': True,
            'aug_reflection': False,
            'aug_dropout': 0.0
        }
        
        augmenter = Augmenter(self.data_config, options)
        
        # Run multiple times to check randomness
        for _ in range(5):
            # Deep copy table to avoid modifying original in loop
            # Note: awkward arrays are immutable-ish but dict is mutable
            table_copy = copy_table(self.table)
            
            augmented = augmenter.augment(table_copy)
            
            # Check shape preserved
            self.assertEqual(len(augmented['pfcand_phirel']), self.n_events)
            
            # Check values changed (rotation applied)
            # Note: random angle could be 0 but unlikely
            # We can check if phirel differences are constant within event (rotation shifts all by same angle)
            
            orig_phi = self.table['pfcand_phirel']
            new_phi = augmented['pfcand_phirel']
            
            # Calculate shift (handling wrapping)
            diff = (new_phi - orig_phi + np.pi) % (2 * np.pi) - np.pi
            
            # Check if diff is constant for all particles in event 0
            diff0 = diff[0]
            self.assertTrue(np.allclose(diff0, diff0[0]))
            
            # Check px, py rotation consistency
            # new_px + i*new_py = (px + i*py) * exp(i*theta)
            # theta = diff
            
            orig_c = self.table['pfcand_px'] + 1j * self.table['pfcand_py']
            new_c = augmented['pfcand_px'] + 1j * augmented['pfcand_py']
            
            # Reconstruct theta from complex rotation
            # new = old * rot => rot = new / old
            # Handle division by zero if any
            
            # Just check magnitude preserved
            self.assertTrue(np.allclose(np.abs(orig_c), np.abs(new_c)))

    def test_reflection(self):
        options = {
            'augment': True,
            'aug_rotation': False,
            'aug_reflection': True,
            'aug_dropout': 0.0
        }
        
        augmenter = Augmenter(self.data_config, options)
        
        # Run multiple times to hit both flip cases
        flipped_phi = False
        flipped_eta = False
        
        for _ in range(20):
            table_copy = copy_table(self.table)
            augmented = augmenter.augment(table_copy)
            
            # Check phi flip
            # If phi flipped, new_phi = -old_phi
            # If not flipped, new_phi = old_phi
            
            # Note: reflection is random per event
            
            # Check event 0
            if np.allclose(augmented['pfcand_phirel'][0], -self.table['pfcand_phirel'][0]):
                flipped_phi = True
                
            # Check eta flip (thetarel)
            if np.allclose(augmented['pfcand_thetarel'][0], -self.table['pfcand_thetarel'][0]):
                flipped_eta = True
                
        # We expect both to happen eventually
        # Wait, my implementation does BOTH independently?
        # Let's check implementation:
        # mask_phi = random choice
        # mask_eta = random choice
        # Yes, independent.
        
        self.assertTrue(flipped_phi, "Phi reflection never happened")
        self.assertTrue(flipped_eta, "Eta reflection never happened")

    def test_dropout(self):
        options = {
            'augment': True,
            'aug_rotation': False,
            'aug_reflection': False,
            'aug_dropout': 0.5 # 50% dropout
        }
        
        augmenter = Augmenter(self.data_config, options)
        
        table_copy = copy_table(self.table)
        augmented = augmenter.augment(table_copy)
        
        # Check mask updated
        orig_mask_sum = ak.sum(self.table['pfcand_mask'])
        new_mask_sum = ak.sum(augmented['pfcand_mask'])
        
        self.assertLess(new_mask_sum, orig_mask_sum, "Dropout didn't remove any particles")
        
        # Check features zeroed out
        # If mask is 0, px should be 0
        
        mask = augmented['pfcand_mask']
        px = augmented['pfcand_px']
        
        # Flatten for easier check
        mask_flat = ak.flatten(mask)
        px_flat = ak.flatten(px)
        
        # Check where mask is 0, px is 0
        self.assertTrue(np.all(px_flat[mask_flat == 0] == 0))

def copy_table(table):
    # Helper to deep copy awkward array dict
    new_table = ak.Array({field: table[field] for field in table.fields})
    return new_table

if __name__ == '__main__':
    unittest.main()
