import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { RouterModule, Routes } from '@angular/router';
import { HeroComponent } from './hero.component';

const routes: Routes = [
  { path: '', component: HeroComponent }
];

@NgModule({
  declarations: [
    HeroComponent
  ],
  imports: [
    CommonModule,
    RouterModule.forChild(routes)
  ]
})
export class HeroModule { } 