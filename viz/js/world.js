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
	css_renderer: null,
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
	log_plot: false,
	active_model: null,
	model_path_prefix:"plot_json/",

	Ammo: null,
	physics: {},
	show_terrain: false,
	show_page: true,
	rotate_plot: false,
	grid: null,
	plot: null,
	origin: new THREE.Vector3(-0.5, 0, -0.5),
	
	page: null,
	page_div: null,
	pages: [],
	pages_scroll: null,
	pages_group: null,
}

const globals = {
}

window.world = world

export { world, globals }