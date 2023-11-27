import * as THREE from 'three'
import { world } from './world.js'
import { CSS3DRenderer } from 'three/examples/jsm/renderers/CSS3DRenderer.js'

function resize_callback() {
	const { camera, renderer, css_renderer } = world
	renderer.setSize(window.innerWidth, window.innerHeight)
	css_renderer.setSize(window.innerWidth, window.innerHeight)
	camera.aspect = window.innerWidth / window.innerHeight
	camera.updateProjectionMatrix()
}

function init_canvas() {
	window.addEventListener('resize', resize_callback)
	
	const container = document.getElementById( 'container' )
	const renderer = new THREE.WebGLRenderer({
		antialias: true,
	})
	renderer.shadowMap.enabled = true
	renderer.shadowMap.type = THREE.PCFSoftShadowMap
	container.appendChild(renderer.domElement)

	renderer.setSize(window.innerWidth, window.innerHeight)
	// renderer.gammaOutput = true
	renderer.gammaFactor = 2.2

	THREE.Cache.enabled = true

	const scene = new THREE.Scene()

	world.renderer = renderer
	world.scene = scene

	// css renderer
	const css_renderer = new CSS3DRenderer()
	css_renderer.setSize(window.innerWidth, window.innerHeight)
	css_renderer.domElement.style.position = 'absolute'
	css_renderer.domElement.style.top = '0px'
	css_renderer.domElement.style.margin = '0px'
	css_renderer.domElement.style.padding = '0px'
	container.appendChild(css_renderer.domElement)
	world.css_renderer = css_renderer
}

export { init_canvas }