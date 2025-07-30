import os
import tempfile
import unittest

import numpy as np

import qhnswlib


class QuantizationTestCase(unittest.TestCase):
    def testInt8QuantizationSpaces(self):
        """Test int8 quantization across all distance spaces"""
        
        dim = 16
        num_elements = 1000
        k = 10
        
        # Generating sample data
        data = np.float32(np.random.random((num_elements, dim)))
        labels = np.arange(num_elements)
        
        # Test all spaces with int8 quantization
        for space_name in ['l2', 'ip', 'cosine']:
            with self.subTest(space=space_name):
                # Create int8 quantized index
                p = qhnswlib.Index(space=space_name, dim=dim, quant='int8')
                p.init_index(max_elements=num_elements, ef_construction=200, M=16)
                p.set_ef(50)
                
                # Add items
                p.add_items(data, labels)
                
                # Test basic properties
                self.assertEqual(p.space, space_name)
                self.assertEqual(p.quant, 'int8')
                self.assertEqual(p.dim, dim)
                self.assertEqual(p.get_current_count(), num_elements)
                
                # Test querying
                query_data = np.float32(np.random.random((5, dim)))
                labels_found, distances = p.knn_query(query_data, k=k)
                
                # Validate output shapes
                self.assertEqual(labels_found.shape, (5, k))
                self.assertEqual(distances.shape, (5, k))
                
                # Validate distance ranges based on space
                if space_name in ['cosine', 'ip']:
                    # Cosine and IP distances can be negative to positive
                    # But should be finite
                    self.assertTrue(np.all(np.isfinite(distances)))
                elif space_name == 'l2':
                    # L2 distances should be non-negative
                    self.assertTrue(np.all(distances >= 0))
                    self.assertTrue(np.all(np.isfinite(distances)))

    def testInt8SerializationCompatibility(self):
        """Test that int8 indexes can be saved and loaded"""
        
        dim = 32
        num_elements = 500
        
        # Generating sample data
        data = np.float32(np.random.random((num_elements, dim)))
        labels = np.arange(num_elements)
        
        for space_name in ['l2', 'cosine']:
            with self.subTest(space=space_name):
                # Create and populate int8 index
                p = qhnswlib.Index(space=space_name, dim=dim, quant='int8')
                p.init_index(max_elements=num_elements, ef_construction=200, M=16)
                p.add_items(data, labels)
                
                # Query before saving
                query_data = np.float32(np.random.random((1, dim)))
                labels_before, distances_before = p.knn_query(query_data, k=5)
                
                # Save index
                with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp_file:
                    index_path = tmp_file.name
                
                try:
                    p.save_index(index_path)
                    
                    # Load index
                    p_loaded = qhnswlib.Index(space=space_name, dim=dim, quant='int8')
                    p_loaded.load_index(index_path, max_elements=num_elements)
                    p_loaded.set_ef(50)
                    
                    # Verify loaded index properties
                    self.assertEqual(p_loaded.space, space_name)
                    self.assertEqual(p_loaded.quant, 'int8')
                    self.assertEqual(p_loaded.dim, dim)
                    self.assertEqual(p_loaded.get_current_count(), num_elements)
                    
                    # Query after loading
                    labels_after, distances_after = p_loaded.knn_query(query_data, k=5)
                    
                    # Results should be identical
                    np.testing.assert_array_equal(labels_before, labels_after)
                    np.testing.assert_array_almost_equal(distances_before, distances_after, decimal=6)
                    
                finally:
                    # Clean up
                    if os.path.exists(index_path):
                        os.unlink(index_path)

    def testInt8VsFloat32Recall(self):
        """Test that int8 quantization maintains reasonable recall vs float32"""
        
        dim = 64
        num_elements = 2000
        k = 20
        
        # Generating sample data
        data = np.float32(np.random.random((num_elements, dim)))
        labels = np.arange(num_elements)
        query_data = np.float32(np.random.random((10, dim)))
        
        for space_name in ['l2', 'cosine']:
            with self.subTest(space=space_name):
                # Create float32 index (baseline)
                p_float = qhnswlib.Index(space=space_name, dim=dim)
                p_float.init_index(max_elements=num_elements, ef_construction=200, M=16)
                p_float.set_ef(100)
                p_float.add_items(data, labels)
                
                # Create int8 index with higher parameters to compensate
                p_int8 = qhnswlib.Index(space=space_name, dim=dim, quant='int8')
                p_int8.init_index(max_elements=num_elements, ef_construction=300, M=24)
                p_int8.set_ef(150)
                p_int8.add_items(data, labels)
                
                # Query both indexes
                labels_float, _ = p_float.knn_query(query_data, k=k)
                labels_int8, _ = p_int8.knn_query(query_data, k=k)
                
                # Calculate recall
                recalls = []
                for i in range(len(query_data)):
                    overlap = len(set(labels_float[i]) & set(labels_int8[i]))
                    recall = overlap / k
                    recalls.append(recall)
                
                mean_recall = np.mean(recalls)
                
                # Int8 should maintain reasonable recall (>70% for synthetic data)
                self.assertGreater(mean_recall, 0.7, 
                                 f"Int8 recall too low for {space_name}: {mean_recall:.3f}")

    def testInt8QuantizationErrors(self):
        """Test error handling for quantization parameters"""
        
        dim = 16
        
        # Test invalid quantization parameter
        with self.assertRaises(RuntimeError):
            p = qhnswlib.Index(space='l2', dim=dim, quant='invalid')
        
        # Test cosine space supports int8 (should not raise error)
        try:
            p = qhnswlib.Index(space='cosine', dim=dim, quant='int8')
            p.init_index(max_elements=100)
        except RuntimeError as e:
            if "not yet supported" in str(e):
                self.fail("Cosine space should support int8 quantization")

    def testInt8MemoryUsage(self):
        """Test that int8 quantization reduces memory usage"""
        
        dim = 128
        num_elements = 1000
        
        # Generate data
        data = np.float32(np.random.random((num_elements, dim)))
        labels = np.arange(num_elements)
        
        # Create float32 index
        p_float = qhnswlib.Index(space='l2', dim=dim)
        p_float.init_index(max_elements=num_elements, M=16)
        p_float.add_items(data, labels)
        
        # Create int8 index
        p_int8 = qhnswlib.Index(space='l2', dim=dim, quant='int8')
        p_int8.init_index(max_elements=num_elements, M=16)
        p_int8.add_items(data, labels)
        
        # Check file sizes (rough memory usage proxy)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp_float:
            float_path = tmp_float.name
        with tempfile.NamedTemporaryFile(delete=False, suffix='.bin') as tmp_int8:
            int8_path = tmp_int8.name
        
        try:
            p_float.save_index(float_path)
            p_int8.save_index(int8_path)
            
            float_size = os.path.getsize(float_path)
            int8_size = os.path.getsize(int8_path)
            
            # Int8 should use significantly less space
            compression_ratio = float_size / int8_size
            self.assertGreater(compression_ratio, 2.0, 
                             f"Int8 compression insufficient: {compression_ratio:.2f}x")
            
        finally:
            # Clean up
            for path in [float_path, int8_path]:
                if os.path.exists(path):
                    os.unlink(path)

    def testInt8WithFiltering(self):
        """Test that int8 quantization works with filtering"""
        
        dim = 32
        num_elements = 1000
        
        # Generate data
        data = np.float32(np.random.random((num_elements, dim)))
        labels = np.arange(num_elements)
        
        # Create int8 index
        p = qhnswlib.Index(space='l2', dim=dim, quant='int8')
        p.init_index(max_elements=num_elements)
        p.add_items(data, labels)
        
        # Define filter (only even labels)
        def even_filter(label):
            return label % 2 == 0
        
        # Query with filter
        query_data = np.float32(np.random.random((1, dim)))
        labels_found, distances = p.knn_query(query_data, k=10, filter=even_filter)
        
        # All returned labels should be even
        for label in labels_found[0]:
            self.assertEqual(label % 2, 0, f"Filter failed: label {label} is not even")



if __name__ == '__main__':
    unittest.main()