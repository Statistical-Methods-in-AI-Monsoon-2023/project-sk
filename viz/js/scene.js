import * as THREE from 'three'
import { world } from './world.js'
import { add_plot, load_model_name } from './plot.js'

function add_ground() {
	const { scene } = world

	const ground = new THREE.Mesh(
		new THREE.PlaneGeometry(500, 500),
		new THREE.MeshBasicMaterial({ color: 'green' }),
	)
	ground.rotation.x = -Math.PI / 2
	ground.position.y = -0.2
	scene.add(ground)
	ground.material.transparent = true
	ground.material.side = THREE.DoubleSide
	ground.material.opacity = 0.4
	ground.castShadow = true
	ground.receiveShadow = true

	world.ground = ground
}


function add_lights() {
	const { scene } = world
	const ambientLight = new THREE.AmbientLight(0xffffff, 0.2)
	scene.add(ambientLight)

	// top light
	const light = new THREE.DirectionalLight(0xffffff, 1)
	light.position.set(-0.5, 2, -0.5)
	scene.add(light)

	// Set up shadow properties for the light
	light.shadow.mapSize.width = 20
	light.shadow.mapSize.height = 20
	light.shadow.camera.near = 1
	light.shadow.camera.far = 10
	light.castShadow = true

	// // add light helper
	// const lightHelper = new THREE.DirectionalLightHelper(light)
	// scene.add(lightHelper)

	// // keep moving light in a circle of radius around (0,5,0)
	// let angle = 0
	// let radius = 0.1
	// const light_move = () => {
	// 	light.position.x = radius * Math.cos(angle)
	// 	light.position.z = radius * Math.sin(angle)
	// 	angle += 0.01
	// }
	// world.light_move = light_move

	// setInterval(light_move, 10)
}

function build_scene() {
	add_ground()
	add_plot()
	load_model_name()
}

export { build_scene, add_lights }