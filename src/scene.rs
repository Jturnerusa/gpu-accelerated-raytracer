use crate::{
    camera::Camera, deserialize, light::Light, material::Material, mesh::Mesh, object::Object,
};

#[derive(Clone, Debug)]
pub struct Scene {
    camera: Camera,
    meshes: Vec<Mesh>,
    objects: Vec<Object>,
    lights: Vec<Light>,
}

impl Scene {
    pub fn camera(&self) -> Camera {
        self.camera
    }

    pub fn meshes(&self) -> &[Mesh] {
        &self.meshes
    }

    pub fn objects(&self) -> &[Object] {
        &self.objects
    }

    pub fn lights(&self) -> &[Light] {
        &self.lights
    }

    pub fn from_deserialized(scene: deserialize::Scene) -> Self {
        let mesh_keys = scene.meshes.keys().collect::<Vec<_>>();

        let meshes = mesh_keys
            .iter()
            .map(|key| scene.meshes[key.as_str()].clone())
            .map(Mesh::from_deserialized)
            .collect::<Vec<_>>();

        let objects = scene
            .objects
            .iter()
            .map(|object| {
                let mesh = mesh_keys
                    .iter()
                    .enumerate()
                    .find_map(|(i, key)| if *object.mesh == **key { Some(i) } else { None })
                    .unwrap();

                Object::new(
                    object.transform,
                    mesh,
                    Material::from_deserialized(scene.materials[object.material.as_str()]),
                )
            })
            .collect::<Vec<_>>();

        let camera = Camera::from_deserialized(scene.camera);

        let lights = scene
            .lights
            .iter()
            .copied()
            .map(Light::from_deserialized)
            .collect::<Vec<_>>();

        Scene {
            camera,
            meshes,
            objects,
            lights,
        }
    }
}
