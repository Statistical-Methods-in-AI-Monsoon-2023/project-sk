import * as THREE from 'three'

const world = {
	debug: '', // debug text
	controls: {
		up: false,
		down: false,
		left: false,
		right: false,
	},
	renderer: null,
	camera: null,
	orbit_cam: null,
	scene: null,
	
	models: {
	},
	textures: {
	},

	plots:{
		loss: {
			data: null,
			mesh: null,
			visible: true,
		},
		acc: {
			data: null,
			mesh: null,
			visible: false,
		}
	},
	model_names: {},
	active_model: null,
	model_path_prefix:"plot_json/"
}

const globals = {
}

export { world, globals }