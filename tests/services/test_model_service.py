import os
import pytest
from unittest.mock import MagicMock, patch
import trimesh
import numpy as np

from heightcraft.services.model_service import ModelService
from heightcraft.core.exceptions import ModelLoadError
from heightcraft.domain.mesh import Mesh

class TestModelService:
    @pytest.fixture
    def mock_cache_manager(self):
        return MagicMock()

    @pytest.fixture
    def model_service(self, mock_cache_manager):
        return ModelService(cache_manager=mock_cache_manager)

    @pytest.fixture
    def mock_mesh(self):
        mesh = MagicMock(spec=trimesh.Trimesh)
        mesh.vertices = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
        mesh.faces = np.array([[0, 1, 2]])
        return mesh

    def test_init(self, mock_cache_manager):
        service = ModelService(cache_manager=mock_cache_manager)
        assert service.cache_manager == mock_cache_manager

    def test_init_default_cache(self):
        with patch('heightcraft.services.model_service.CacheManager') as MockCache:
            service = ModelService()
            assert service.cache_manager == MockCache.return_value

    def test_load_model_not_found(self, model_service):
        with pytest.raises(ModelLoadError, match="Model file not found"):
            model_service.load_model("non_existent_file.stl")

    def test_load_model_from_cache(self, model_service, mock_cache_manager, mock_mesh):
        file_path = "test.stl"
        cached_mesh = Mesh(mock_mesh)
        
        with patch('os.path.exists', return_value=True):
            mock_cache_manager.has.return_value = True
            mock_cache_manager.get.return_value = cached_mesh
            
            result = model_service.load_model(file_path, use_cache=True)
            
            assert result == cached_mesh
            mock_cache_manager.get.assert_called_once_with(file_path)

    def test_load_model_trimesh_load(self, model_service, mock_cache_manager, mock_mesh):
        file_path = "test.stl"
        
        with patch('os.path.exists', return_value=True), \
             patch('trimesh.load', return_value=mock_mesh) as mock_load:
            
            mock_cache_manager.has.return_value = False
            
            result = model_service.load_model(file_path, use_cache=True)
            
            assert isinstance(result, Mesh)
            assert result.mesh == mock_mesh
            mock_load.assert_called_once_with(file_path)
            mock_cache_manager.set.assert_called_once()

    def test_load_model_scene(self, model_service, mock_mesh):
        file_path = "test.glb"
        mock_scene = MagicMock(spec=trimesh.Scene)
        mock_scene.geometry = {'mesh1': mock_mesh}
        
        with patch('os.path.exists', return_value=True), \
             patch('trimesh.load', return_value=mock_scene):
            
            result = model_service.load_model(file_path, use_cache=False)
            
            assert isinstance(result, Mesh)
            assert result.mesh == mock_mesh

    def test_load_model_scene_multiple_meshes(self, model_service, mock_mesh):
        file_path = "test.glb"
        mock_scene = MagicMock(spec=trimesh.Scene)
        mock_mesh2 = MagicMock(spec=trimesh.Trimesh)
        mock_scene.geometry = {'mesh1': mock_mesh, 'mesh2': mock_mesh2}
        
        combined_mesh = MagicMock(spec=trimesh.Trimesh)
        combined_mesh.vertices = np.array([[0, 0, 0]])
        combined_mesh.faces = np.array([[0]])
        
        with patch('os.path.exists', return_value=True), \
             patch('trimesh.load', return_value=mock_scene), \
             patch('trimesh.util.concatenate', return_value=combined_mesh) as mock_concat, \
             patch('heightcraft.domain.mesh.Mesh._validate_mesh'): # Mock validation method
            
            result = model_service.load_model(file_path, use_cache=False)
            
            assert isinstance(result, Mesh)
            assert result.mesh == combined_mesh
            mock_concat.assert_called_once()

    def test_load_model_unsupported_format(self, model_service):
        file_path = "test.txt"
        with patch('os.path.exists', return_value=True):
            # We need to ensure use_cache doesn't bypass the format check
            # But the default is use_cache=True.
            # If cache miss, it proceeds to format check.
            # So we need to mock cache miss.
            model_service.cache_manager.has.return_value = False
            
            with pytest.raises(ModelLoadError, match="Unsupported file format"):
                model_service.load_model(file_path)

    def test_load_model_exception(self, model_service):
        file_path = "test.stl"
        with patch('os.path.exists', return_value=True), \
             patch('trimesh.load', side_effect=Exception("Load error")):
            
            model_service.cache_manager.has.return_value = False
            
            with pytest.raises(ModelLoadError, match="Failed to load model"):
                model_service.load_model(file_path)

    def test_get_supported_formats(self, model_service):
        formats = model_service.get_supported_formats()
        assert '.stl' in formats
        assert '.obj' in formats
        assert '.ply' in formats

    def test_convert_model_format(self, model_service, mock_mesh):
        mesh = Mesh(mock_mesh)
        output_path = "output.obj"
        
        with patch.object(mock_mesh, 'export') as mock_export:
            result = model_service.convert_model_format(mesh, '.obj', output_path)
            
            assert result == output_path
            mock_export.assert_called_once_with(output_path)

    def test_convert_model_format_add_extension(self, model_service, mock_mesh):
        mesh = Mesh(mock_mesh)
        output_path = "output"
        
        with patch.object(mock_mesh, 'export') as mock_export:
            result = model_service.convert_model_format(mesh, '.obj', output_path)
            
            assert result == "output.obj"
            mock_export.assert_called_once_with("output.obj")

    def test_convert_model_format_unsupported(self, model_service, mock_mesh):
        mesh = Mesh(mock_mesh)
        with pytest.raises(ModelLoadError, match="Unsupported output format"):
            model_service.convert_model_format(mesh, '.txt', "output.txt")

    def test_convert_model_format_error(self, model_service, mock_mesh):
        mesh = Mesh(mock_mesh)
        with patch.object(mock_mesh, 'export', side_effect=Exception("Export error")):
            with pytest.raises(ModelLoadError, match="Failed to convert model"):
                model_service.convert_model_format(mesh, '.obj', "output.obj")
