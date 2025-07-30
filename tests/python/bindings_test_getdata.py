import unittest

import numpy as np

import qhnswlib


class RandomSelfTestCase(unittest.TestCase):
    def testGettingItems(self):
        print("\n**** Getting the data by label test ****\n")

        dim = 16
        num_elements = 10000

        # Generating sample data
        data = np.float32(np.random.random((num_elements, dim)))
        labels = np.arange(0, num_elements)

        # Test both float32 and int8 quantized indexes
        for quant_type in [None, 'int8']:
            print(f"Testing with quantization: {quant_type}")
            
            # Declaring index
            if quant_type:
                p = qhnswlib.Index(space='l2', dim=dim, quant=quant_type)
            else:
                p = qhnswlib.Index(space='l2', dim=dim)

            p.init_index(max_elements=num_elements, ef_construction=100, M=16)
            p.set_ef(100)
            p.set_num_threads(4)

            # Before adding anything, getting any labels should fail
            self.assertRaises(Exception, lambda: p.get_items(labels))

            print("Adding all elements (%d)" % (len(data)))
            p.add_items(data, labels)

            # Getting data by label should raise an exception if a scalar is passed:
            self.assertRaises(ValueError, lambda: p.get_items(labels[0]))

            # After adding them, all labels should be retrievable
            returned_items_np = p.get_items(labels)
            
            # Test that no NaN values are returned (this was the main bug)
            self.assertFalse(np.isnan(returned_items_np).any(), 
                           f"NaN values found in get_items for {quant_type}")
            
            if quant_type == 'int8':
                # For int8, expect approximate equality due to quantization error
                max_error = np.max(np.abs(data - returned_items_np))
                print(f"Max quantization error: {max_error}")
                self.assertLess(max_error, 0.1, "Quantization error too large")
                
                # Test get_quantized_items method
                test_labels = labels[:10]  # Test first 10 items
                quantized_data = p.get_quantized_items(test_labels)
                self.assertEqual(len(quantized_data), len(test_labels))
                
                for i, (vec, scale) in enumerate(quantized_data):
                    self.assertEqual(vec.dtype, np.int8)
                    self.assertEqual(vec.shape, (dim,))
                    self.assertIsInstance(scale, float)
                    self.assertGreater(scale, 0)
            else:
                # For float32, expect exact equality
                self.assertTrue((data == returned_items_np).all())

            # check returned type of get_items
            self.assertTrue(isinstance(returned_items_np, np.ndarray))
            returned_items_list = p.get_items(labels, return_type="list")
            self.assertTrue(isinstance(returned_items_list, list))
            self.assertTrue(isinstance(returned_items_list[0], list))
