import * as THREE from 'three'
import { world } from './world.js'
import { add_plot, load_model_name } from './plot.js'
import { CSS3DObject } from 'three/examples/jsm/renderers/CSS3DRenderer.js'
import * as TWEEN from 'tween'

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

function add_skybox() {
	const loader = new THREE.CubeTextureLoader()
	const texture = loader.load([
		'./skybox/back.png',
		'./skybox/front.png',
		'./skybox/top.png',
		'./skybox/bottom.png',
		'./skybox/right.png',
		'./skybox/left.png',
	])
	world.scene.background = texture
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

function add_page(src) {
	const div = document.createElement('div')
	div.style.width = '768px'
	div.style.height = '500px'
	div.style.backgroundColor = 'transparent'

	const iframe = document.createElement('iframe')
	iframe.style.width = '768px'
	iframe.style.height = '500px'
	iframe.style.border = '0px'
	iframe.src = src || 'https://www.google.com'
	iframe.allowTransparency = true
	div.appendChild(iframe)

	const page = new CSS3DObject(div)

	world.page = page
	world.page_div = div

	world.pages.push(page)
	world.pages_group.add(page)

	page.scale.set(0.005, 0.005, 0.005)
	// page.position.set(-2, 1, 0)

	// calculate page position
	page.position.set(-2.5, world.pages.length * 4 - 3, 0)
}

function pages_scroll(p) {
	// animate page position
	const scroll_dist = p * 4
	for (let page of world.pages) {
		const startPosition = page.position.y;
		const targetPosition = startPosition + scroll_dist;
		new TWEEN.Tween(page.position)
			.easing(TWEEN.Easing.Back.InOut)
			.to({ y: targetPosition }, 750)
			.start();
	}
}

world.pages_scroll = pages_scroll

function build_scene() {

	const { scene } = world
	const pages_group = new THREE.Group()
	scene.add(pages_group)
	world.pages_group = pages_group

	// add_ground()
	add_skybox()
	add_plot()
	load_model_name()
	add_page("./slides/blog.html")
	add_page("./slides/blog.html")
	add_page("./slides/blog.html")
	add_page("./slides/blog.html")
	add_page("./slides/blog.html")
	add_page("./slides/blog.html")
}

export { build_scene, add_lights }