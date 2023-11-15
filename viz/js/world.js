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
	multiple_plots: false,
	blend_plot: {
		data: null,
		processed: null,
	},
	plots:{
		loss: {
			data: null,
			processed: null,
			visible: true,
		},
		acc: {
			data: null,
			processed: null,
			visible: false,
		}
	},
	model_names: {},
	active_model: null,
	model_path_prefix:"plot_json/"
}

const globals = {
}

window.world = world

export { world, globals }