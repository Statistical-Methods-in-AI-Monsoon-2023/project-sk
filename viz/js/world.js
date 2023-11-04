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
	}
}

const globals = {
}

export { world, globals }